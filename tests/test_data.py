from pathlib import Path
from typing import Any

import pytest

from helixdepth.data import canonical_text, select_documents, split_for_hash, text_hash


def settings(seed: int = 2026) -> dict[str, Any]:
    return {"text_column": "text", "seed": seed, "candidate_documents": 100,
            "max_documents": 20, "max_text_bytes": 10000, "min_document_bytes": 1,
            "max_document_bytes": 500, "validation_per_mille": 500,
            "excluded_domains": ["wikipedia.org"]}


def source_rows() -> list[dict[str, Any]]:
    return [{"id": str(index), "text": f"Document {index} has distinct content.",
             "url": "https://example.com", "dump": "test"} for index in range(50)]


def test_duplicate_text_grouping_before_split() -> None:
    rows = source_rows()
    rows.extend([{"id": "duplicate", "text": "\r\nDocument 0 has distinct content.\r\n"},
                 {"id": "unicode-a", "text": "café"},
                 {"id": "unicode-b", "text": "cafe\u0301"}])
    config = settings()
    config["max_documents"] = 100
    selected, counts = select_documents(rows, config)
    assert counts["duplicate_documents"] == 2
    assert len(selected) == 51
    zero = next(row for row in selected if row["text"] == rows[0]["text"])
    assert {source["id"] for source in zero["sources"]} == {"0", "duplicate"}
    unicode_document = next(row for row in selected if row["text"] == "café")
    assert len(unicode_document["sources"]) == 2
    hashes = {split: {row["text_sha256"] for row in selected if row["split"] == split}
              for split in ("train", "validation")}
    assert not hashes["train"] & hashes["validation"]


def test_selection_reproducibility_seed_and_caps() -> None:
    first, counts = select_documents(source_rows(), settings())
    repeated, repeated_counts = select_documents(source_rows(), settings())
    assert first == repeated and counts == repeated_counts
    changed, _ = select_documents(source_rows(), settings(seed=7))
    assert [row["document_id"] for row in first] != [row["document_id"] for row in changed]
    config = settings()
    config["max_text_bytes"] = 300
    bounded, _ = select_documents(source_rows(), config)
    assert len(bounded) <= config["max_documents"]
    assert sum(row["text_bytes"] for row in bounded) <= 300
    assert all(row["split"] == split_for_hash(row["text_sha256"], 2026, 500) for row in bounded)


def test_domain_and_length_filters_and_canonical_hash() -> None:
    rows = source_rows()
    rows[0]["url"] = "https://en.wikipedia.org/wiki/Test"
    rows[1]["text"] = "x" * 501
    config = settings()
    config["candidate_documents"] = 40
    selected, counts = select_documents(rows, config)
    assert counts["scanned_documents"] == 40
    assert counts["excluded_domain"] == counts["excluded_length"] == 1
    assert all(source["id"] not in {"0", "1"} for row in selected for source in row["sources"])
    assert text_hash(canonical_text(" café\r\n")) == text_hash(canonical_text("cafe\u0301"))


def test_tokenizer_fit_consumes_only_training_ids(tmp_path: Path) -> None:
    from scripts.prepare_cp2 import fit_tokenizer, write_rows

    train_path = tmp_path / "train.jsonl"
    rows = [{"document_id": str(index), "text": f"Document {index}: learning byte combinations and English words."}
            for index in range(100)]
    write_rows(train_path, iter(rows))
    write_rows(tmp_path / "validation.jsonl", iter([{"document_id": "held-out", "text": "never fitted"}]))
    config = {"vocab_size": 300, "min_pair_frequency": 1,
              "special_tokens": ["<pad>", "<bos>", "<eos>", "<unk>"]}
    tokenizer, fitted_ids = fit_tokenizer(train_path, config)
    assert fitted_ids == [row["document_id"] for row in rows]
    assert "held-out" not in fitted_ids
    assert tokenizer.get_vocab_size(with_added_tokens=True) == 300
    write_rows(tmp_path / "validation.jsonl", iter([{"document_id": "changed-held-out", "text": "changed"}]))
    repeated, repeated_ids = fit_tokenizer(train_path, config)
    assert repeated_ids == fitted_ids
    assert repeated.to_str() == tokenizer.to_str()


def test_saved_split_verification_rejects_tampering(tmp_path: Path) -> None:
    from scripts.prepare_cp2 import check_documents, file_hash, write_rows

    config = settings()
    documents, _ = select_documents(source_rows(), config)
    for split in ("train", "validation"):
        write_rows(tmp_path / f"{split}.jsonl", (row for row in documents if row["split"] == split))
    (tmp_path / "source_card.md").write_text("fixture", encoding="utf-8")
    manifest = {"settings": config, "files": {f"{split}.jsonl": file_hash(tmp_path / f"{split}.jsonl")
                for split in ("train", "validation")}, "source_card_sha256": file_hash(tmp_path / "source_card.md")}
    check_documents(tmp_path, manifest)
    with (tmp_path / "validation.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        check_documents(tmp_path, manifest)
