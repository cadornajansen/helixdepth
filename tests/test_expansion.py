import io
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers

from helixdepth.data import split_for_hash, text_hash
from helixdepth.expansion import (encode_document, expand_corpus, new_documents,
                                 read_rows, verify_expansion, write_json, write_row)
from helixdepth.packing import PackedStream, sha256_file

Corpus = tuple[Path, Path, list[dict[str, Any]], dict[str, int]]


@pytest.fixture
def corpus(tmp_path: Path) -> Corpus:
    base = tmp_path / "base"
    base.mkdir()
    vocab = {word: index for index, word in enumerate(
        ["<pad>", "<bos>", "<eos>", "<unk>"] + sorted(pre_tokenizers.ByteLevel.alphabet()))}
    for index in range(len(vocab), 16384):
        vocab[f"unused_piece_{index}"] = index
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.save(str(base / "tokenizer.json"))
    settings = {"seed": 2026, "validation_per_mille": 300, "text_column": "text",
                "min_document_bytes": 1, "max_document_bytes": 200,
                "excluded_domains": ["wikipedia.org"], "revision": "fixture-revision"}
    rows = [{"id": str(index), "text": f"English document number {index}, café.",
             "url": "https://example.com", "dump": "fixture"} for index in range(80)]
    stats: dict[str, Any] = {}
    for split in ("train", "validation"):
        candidates = [row for row in rows if split_for_hash(text_hash(row["text"]), 2026, 300) == split][:3]
        tokens = text_bytes = 0
        with (base / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for row in candidates:
                digest = text_hash(row["text"])
                document = {"document_id": digest, "text_sha256": digest, "text": row["text"],
                            "text_bytes": len(row["text"].encode()), "split": split,
                            "sources": [{"id": row["id"]}]}
                write_row(stream, document)
                tokens += len(encode_document(document, tokenizer))
                text_bytes += document["text_bytes"]
        stats[split] = {"documents": 3, "tokens_including_eos": tokens,
                        "text_tokens": tokens - 3, "text_bytes": text_bytes}
    write_json(base / "tokenizer_training.json", {"fixture": True, "training_split": "train"})
    (base / "source_card.md").write_text("Fixture provenance", encoding="utf-8")
    evidence = tmp_path / "evidence.json"
    write_json(evidence, {"passed": True, "vocab_size_including_special_tokens": 16384,
                         "settings": settings, "splits": stats,
                         "files": {p.name: {"sha256": sha256_file(p)} for p in base.iterdir()}})
    limits = {"candidate_documents": 80, "max_train_tokens": 2000,
              "max_train_documents": 60, "max_train_text_bytes": 3000}
    return base, evidence, rows, limits


def test_expansion_reproduces_and_preserves_frozen_artifacts(corpus: Corpus, tmp_path: Path) -> None:
    base, evidence, rows, limits = corpus
    first, second = tmp_path / "first", tmp_path / "second"
    expand_corpus(rows, base, evidence, first, limits)
    expand_corpus(rows, base, evidence, second, limits)
    assert {p.name: sha256_file(p) for p in first.iterdir()} == {p.name: sha256_file(p) for p in second.iterdir()}
    report = verify_expansion(first, base, evidence)
    assert report["passed"] and report["new_training_documents"] > 0
    for name in ("tokenizer.json", "validation.jsonl", "tokenizer_training.json"):
        assert (first / name).read_bytes() == (base / name).read_bytes()
    original = list(read_rows(base / "train.jsonl"))
    assert list(read_rows(first / "train.jsonl"))[:len(original)] == original
    stream = PackedStream(first, "train", 512, 2)
    inputs, targets = stream.peek()
    assert inputs.shape == targets.shape == (2, 512)
    assert inputs.flatten()[1:].equal(targets.flatten()[:-1])
    assert stream.peek()[0].equal(PackedStream(first, "train", 512, 2).peek()[0])


def test_selection_excludes_held_out_duplicate_ids_text_and_domains(corpus: Corpus) -> None:
    base, evidence, rows, _ = corpus
    settings = json.loads(evidence.read_text())["settings"]
    heldout = next(read_rows(base / "validation.jsonl"))
    extra = [{"id": "alias-heldout", "text": "\r\n" + heldout["text"] + "\r\n"}]
    extra += [{"id": "duplicate-alias", "text": rows[40]["text"]}]
    extra += [{"id": "bad-domain", "text": "Wikipedia entry here", "url": "https://en.wikipedia.org/wiki/Test"}]
    counts: Counter[str] = Counter()
    audit = io.StringIO()
    selected = list(new_documents(rows + extra, base, settings, 100, audit, counts))
    hashes = [row["text_sha256"] for row in selected]
    assert len(hashes) == len(set(hashes))
    assert heldout["text_sha256"] not in hashes
    assert all(split_for_hash(digest, 2026, 300) == "train" for digest in hashes)
    assert not any(source["id"] == "bad-domain" for row in selected for source in row["sources"])
    assert counts["duplicate_or_existing"] >= 8
    assert counts["reserved_validation"] > 0


def test_conflicting_frozen_source_id_fails_closed(corpus: Corpus) -> None:
    base, evidence, _, _ = corpus
    heldout = next(read_rows(base / "validation.jsonl"))
    settings = json.loads(evidence.read_text())["settings"]
    with pytest.raises(ValueError, match="conflicting content"):
        list(new_documents([{"id": heldout["sources"][0]["id"], "text": "different text"}],
                           base, settings, 10, io.StringIO(), Counter()))


def test_capacity_stops_at_whole_document_and_never_overwrites(corpus: Corpus, tmp_path: Path) -> None:
    base, evidence, rows, limits = corpus
    limits["max_train_tokens"] = 300
    out = tmp_path / "bounded"
    manifest = expand_corpus(rows, base, evidence, out, limits)
    assert manifest["splits"]["train"]["tokens"] <= 300
    assert manifest["splits"]["train"]["stop_reason"] == "first_document_exceeding_capacity"
    assert verify_expansion(out, base, evidence)["passed"]
    with pytest.raises(FileExistsError):
        expand_corpus(rows, base, evidence, out, limits)


def test_verification_reencodes_tokens_even_if_file_hash_updated(corpus: Corpus, tmp_path: Path) -> None:
    base, evidence, rows, limits = corpus
    out = tmp_path / "tampered"
    manifest = expand_corpus(rows, base, evidence, out, limits)
    with (out / "train.bin").open("r+b") as stream:
        stream.write(b"\x00\x00\x00\x00")
    manifest["files"]["train.bin"]["sha256"] = sha256_file(out / "train.bin")
    write_json(out / "manifest.json", manifest)
    with pytest.raises(ValueError, match="Packed IDs differ"):
        verify_expansion(out, base, evidence)


def test_modified_tokenizer_or_base_cannot_be_used(corpus: Corpus, tmp_path: Path) -> None:
    base, evidence, rows, limits = corpus
    with (base / "tokenizer.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="Frozen CP2 checksum"):
        expand_corpus(rows, base, evidence, tmp_path / "bad", limits)
    assert not (tmp_path / "bad").exists()


def test_changed_source_audit_order_is_rejected(corpus: Corpus, tmp_path: Path) -> None:
    base, evidence, rows, limits = corpus
    out = tmp_path / "audit-tampered"
    manifest = expand_corpus(rows, base, evidence, out, limits)
    path = out / "selection_audit.jsonl"
    audit = list(read_rows(path))
    audit[0]["source"]["source_index"] = 9
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in audit:
            write_row(stream, row)
    manifest["files"][path.name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    write_json(out / "manifest.json", manifest)
    with pytest.raises(ValueError, match="contiguous prefix"):
        verify_expansion(out, base, evidence)


@pytest.mark.parametrize("bound,value", [("candidate_documents", 20), ("max_train_documents", 8),
                                         ("max_train_text_bytes", 300)])
def test_independent_preparation_bounds(corpus: Corpus, tmp_path: Path, bound: str, value: int) -> None:
    base, evidence, rows, limits = corpus
    limits[bound] = value
    out = tmp_path / bound
    manifest = expand_corpus(rows, base, evidence, out, limits)
    assert verify_expansion(out, base, evidence)["passed"]
    assert manifest["selection_counts"]["scanned_documents"] <= limits["candidate_documents"]
    assert manifest["splits"]["train"]["documents"] <= limits["max_train_documents"]
    assert manifest["splits"]["train"]["text_bytes"] <= limits["max_train_text_bytes"]
