"""Extend a frozen corpus without fitting a tokenizer or changing validation."""
import hashlib
import importlib.metadata
import itertools
import json
import shutil
import sys
from array import array
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, TextIO
from urllib.parse import urlsplit

from tokenizers import Tokenizer

from helixdepth.data import canonical_text, split_for_hash, text_hash
from helixdepth.packing import sha256_file


def read_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def write_row(stream: TextIO, row: dict[str, Any]) -> None:
    stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def checked_base(base: Path, evidence: Path) -> dict[str, Any]:
    report = json.loads(evidence.read_text(encoding="utf-8"))
    if not report["passed"] or report["vocab_size_including_special_tokens"] != 16384:
        raise ValueError("Expected verified CP2 evidence with vocabulary 16384")
    for name, info in report["files"].items():
        if sha256_file(base / name) != info["sha256"]:
            raise ValueError(f"Frozen CP2 checksum mismatch: {name}")
    return report


def new_documents(rows: Iterable[dict[str, Any]], base: Path, settings: dict[str, Any],
                  candidate_limit: int, audit: TextIO, counts: Counter[str]
                  ) -> Iterator[dict[str, Any]]:
    """First occurrence wins; reserve every hash-assigned held-out group."""
    seen_hashes: set[str] = set()
    seen_ids: dict[str, str] = {}
    for split in ("train", "validation"):
        for document in read_rows(base / f"{split}.jsonl"):
            seen_hashes.add(document["text_sha256"])
            for source in document["sources"]:
                seen_ids[source["id"]] = document["text_sha256"]
    for index, row in enumerate(itertools.islice(rows, candidate_limit)):
        raw = row[settings["text_column"]]
        if not isinstance(raw, str):
            raise ValueError("Source text must be a string")
        text = canonical_text(raw)
        digest = text_hash(text)
        source_id = str(row["id"])
        size = len(text.encode("utf-8"))
        host = (urlsplit(str(row.get("url", ""))).hostname or "").lower()
        split = split_for_hash(digest, settings["seed"], settings["validation_per_mille"])
        if source_id in seen_ids and seen_ids[source_id] != digest:
            raise ValueError("One source ID has conflicting content, possibly frozen validation")
        if source_id in seen_ids or digest in seen_hashes:
            reason = "duplicate_or_existing"
        elif split == "validation":
            reason = "reserved_validation"
        elif not settings["min_document_bytes"] <= size <= settings["max_document_bytes"]:
            reason = "excluded_length"
        elif any(host == domain or host.endswith("." + domain)
                 for domain in settings["excluded_domains"]):
            reason = "excluded_domain"
        else:
            reason = "eligible_train"
        seen_hashes.add(digest)
        seen_ids[source_id] = digest
        counts["scanned_documents"] += 1
        counts[reason] += 1
        source = {"id": source_id, "url": row.get("url"), "dump": row.get("dump"),
                  "source_index": index, "raw_text_sha256": text_hash(raw)}
        write_row(audit, {"source": source, "text_sha256": digest,
                          "text_bytes": size, "assigned_split": split, "decision": reason})
        if counts["scanned_documents"] % 10000 == 0:
            print(f"Scanned {counts['scanned_documents']:,} source documents", flush=True)
        if reason == "eligible_train":
            yield {"document_id": digest, "text_sha256": digest, "text": text,
                   "text_bytes": size, "sources": [source], "split": "train"}


def encode_document(document: dict[str, Any], tokenizer: Tokenizer) -> list[int]:
    ids = tokenizer.encode(document["text"], add_special_tokens=False).ids
    if 3 in ids or tokenizer.decode(ids, skip_special_tokens=False) != document["text"]:
        raise ValueError("Frozen tokenizer failed exact document round trip")
    return ids + [2]


def write_split(documents: Iterable[dict[str, Any]], output: Path, split: str,
                tokenizer: Tokenizer, limits: dict[str, int] | None = None
                ) -> dict[str, Any]:
    tokens = text_bytes = count = 0
    stop_reason = "source_bound_or_exhaustion"
    with (output / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n") as texts, \
            (output / f"{split}.index.jsonl").open("w", encoding="utf-8", newline="\n") as index, \
            (output / f"{split}.bin").open("wb") as binary:
        for document in documents:
            ids = encode_document(document, tokenizer)
            if limits and (tokens + len(ids) > limits["max_train_tokens"] or
                           count + 1 > limits["max_train_documents"] or
                           text_bytes + document["text_bytes"] > limits["max_train_text_bytes"]):
                stop_reason = "first_document_exceeding_capacity"
                break
            write_row(texts, document)
            write_row(index, {"document_id": document["document_id"], "offset": tokens,
                              "tokens": len(ids)})
            array("i", ids).tofile(binary)
            tokens += len(ids)
            count += 1
            text_bytes += document["text_bytes"]
            if count % 10000 == 0:
                print(f"Saved {count:,} {split} documents / {tokens:,} tokens", flush=True)
    return {"tokens": tokens, "text_tokens": tokens - count, "documents": count,
            "text_bytes": text_bytes, "stop_reason": stop_reason,
            "bin_sha256": sha256_file(output / f"{split}.bin"),
            "index_sha256": sha256_file(output / f"{split}.index.jsonl")}


def expand_corpus(rows: Iterable[dict[str, Any]], base: Path, evidence: Path,
                  output: Path, limits: dict[str, int]) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Preserving existing corpus: {output}; choose a new output")
    if sys.byteorder != "little" or array("i").itemsize != 4:
        raise RuntimeError("Packing requires little-endian 32-bit integers")
    report = checked_base(base, evidence)
    original = report["splits"]["train"]
    if any(type(value) is not int or value < 1 for value in limits.values()):
        raise ValueError("Expansion limits must be positive integers")
    if (limits["max_train_tokens"] <= original["tokens_including_eos"] or
            limits["max_train_documents"] <= original["documents"] or
            limits["max_train_text_bytes"] <= original["text_bytes"]):
        raise ValueError("Expansion limits must preserve and extend the original training split")
    tokenizer = Tokenizer.from_file(str(base / "tokenizer.json"))
    if tokenizer.get_vocab_size(with_added_tokens=True) != 16384:
        raise ValueError("Frozen vocabulary changed")
    output.mkdir(parents=True)
    for name in ("tokenizer.json", "tokenizer_training.json", "source_card.md"):
        shutil.copyfile(base / name, output / name)
    counts: Counter[str] = Counter()
    with (output / "selection_audit.jsonl").open("w", encoding="utf-8", newline="\n") as audit:
        additions = new_documents(rows, base, report["settings"], limits["candidate_documents"], audit, counts)
        train = write_split(itertools.chain(read_rows(base / "train.jsonl"), additions),
                            output, "train", tokenizer, limits)
    if train["documents"] <= original["documents"]:
        raise ValueError("No new training documents selected; incomplete output preserved")
    validation = write_split(read_rows(base / "validation.jsonl"), output, "validation", tokenizer)
    # Preserve the primary validation text file byte-for-byte, not just semantically.
    shutil.copyfile(base / "validation.jsonl", output / "validation.jsonl")
    files = {path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
             for path in sorted(output.iterdir()) if path.is_file()}
    manifest = {
        "format": "little-endian-int32", "vocab_size": 16384, "eos_id": 2,
        "tokenizer_sha256": report["files"]["tokenizer.json"]["sha256"],
        "source_revision": report["settings"]["revision"], "source_settings": report["settings"],
        "limits": limits, "selection_counts": dict(counts), "files": files,
        "splits": {"train": train, "validation": validation},
        "base_evidence_sha256": sha256_file(evidence),
        "base_files": report["files"], "original_train_documents": original["documents"],
        "order": "Original CP2 training order, then first eligible occurrence in pinned source order; no shuffle or cycling",
        "selection": "Scan at most candidate_documents rows; same canonicalization, seed, split, length/domain filters as CP2. Exclude all existing and newly seen hashes/IDs. Stop before first whole document exceeding a capacity. Audit may contain that final unselected eligible row.",
        "held_out_policy": "Original validation unchanged; additional hash-assigned validation groups reserved in audit, never trained or added to primary validation",
        "packing": "One EOS per document; contiguous next-token windows with stride=context; same stream for both models",
        "versions": {name: importlib.metadata.version(name) for name in ("datasets", "tokenizers")},
        "code_sha256": {"expansion.py": sha256_file(Path(__file__)),
                        "data.py": sha256_file(Path(__file__).with_name("data.py"))},
        "limitations": ["Bounded source prefix, not a uniform sample", "Canonical exact deduplication only; near duplicates untested", "Domain exclusions do not prove benchmark decontamination", "Capacity is not the final training token budget"],
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def verify_selection(output: Path, base: Path, manifest: dict[str, Any]) -> None:
    settings = manifest["source_settings"]
    hashes: set[str] = set()
    ids: dict[str, str] = {}
    for split in ("train", "validation"):
        for document in read_rows(base / f"{split}.jsonl"):
            hashes.add(document["text_sha256"])
            for source in document["sources"]:
                ids[source["id"]] = document["text_sha256"]
    additions = itertools.islice(read_rows(output / "train.jsonl"), manifest["original_train_documents"], None)
    pending = next(additions, None)
    counts: Counter[str] = Counter()
    unselected_eligible = 0
    for index, row in enumerate(read_rows(output / "selection_audit.jsonl")):
        source, digest = row["source"], row["text_sha256"]
        if source["source_index"] != index:
            raise ValueError("Source audit is not a contiguous prefix")
        if source["id"] in ids and ids[source["id"]] != digest:
            raise ValueError("Conflicting source ID in audit")
        split = split_for_hash(digest, settings["seed"], settings["validation_per_mille"])
        host = (urlsplit(str(source.get("url", ""))).hostname or "").lower()
        if digest in hashes or source["id"] in ids:
            reason = "duplicate_or_existing"
        elif split == "validation":
            reason = "reserved_validation"
        elif not settings["min_document_bytes"] <= row["text_bytes"] <= settings["max_document_bytes"]:
            reason = "excluded_length"
        elif any(host == domain or host.endswith("." + domain) for domain in settings["excluded_domains"]):
            reason = "excluded_domain"
        else:
            reason = "eligible_train"
        if row["assigned_split"] != split or row["decision"] != reason:
            raise ValueError("Audit selection decision differs from policy")
        ids[source["id"]] = digest
        hashes.add(digest)
        counts["scanned_documents"] += 1
        counts[reason] += 1
        if reason == "eligible_train":
            if pending is None:
                unselected_eligible += 1
            else:
                if pending["text_sha256"] != digest or pending["sources"] != [source]:
                    raise ValueError("New training order/provenance differs from source audit")
                pending = next(additions, None)
    if (pending is not None or unselected_eligible > 1 or dict(counts) != manifest["selection_counts"] or
            counts["scanned_documents"] > manifest["limits"]["candidate_documents"]):
        raise ValueError("Source selection counts, bounds or retained documents differ")


def verify_expansion(output: Path, base: Path, evidence: Path) -> dict[str, Any]:
    report = checked_base(base, evidence)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if manifest["base_evidence_sha256"] != sha256_file(evidence):
        raise ValueError("Original evidence identity changed")
    if manifest["source_settings"] != report["settings"] or manifest["source_revision"] != report["settings"]["revision"]:
        raise ValueError("Pinned source settings changed")
    if manifest["original_train_documents"] != report["splits"]["train"]["documents"]:
        raise ValueError("Original training document count changed")
    for name, info in manifest["files"].items():
        if sha256_file(output / name) != info["sha256"] or (output / name).stat().st_size != info["bytes"]:
            raise ValueError(f"Expanded file checksum mismatch: {name}")
    for name in ("tokenizer.json", "tokenizer_training.json", "validation.jsonl", "source_card.md"):
        if sha256_file(output / name) != report["files"][name]["sha256"]:
            raise ValueError(f"Frozen artifact changed: {name}")
    tokenizer = Tokenizer.from_file(str(output / "tokenizer.json"))
    if tokenizer.get_vocab_size(with_added_tokens=True) != 16384 or manifest["tokenizer_sha256"] != sha256_file(output / "tokenizer.json"):
        raise ValueError("Vocabulary/tokenizer identity mismatch")
    hashes: set[str] = set()
    source_ids: set[str] = set()
    base_train = iter(read_rows(base / "train.jsonl"))
    statistics: dict[str, Any] = {}
    for split in ("validation", "train"):
        tokens = count = text_bytes = 0
        rebuilt = hashlib.sha256()
        with (output / f"{split}.bin").open("rb") as binary:
            for document, index in itertools.zip_longest(read_rows(output / f"{split}.jsonl"),
                                                         read_rows(output / f"{split}.index.jsonl")):
                if document is None or index is None:
                    raise ValueError("Document/index row count mismatch")
                digest = text_hash(document["text"])
                settings = report["settings"]
                if (digest != document["document_id"] or digest != document["text_sha256"] or
                        digest in hashes or document["split"] != split or
                        split_for_hash(digest, settings["seed"], settings["validation_per_mille"]) != split):
                    raise ValueError("Duplicate, invalid document hash or split leakage")
                hashes.add(digest)
                if not document["sources"]:
                    raise ValueError("Missing document provenance")
                for source in document["sources"]:
                    if source["id"] in source_ids:
                        raise ValueError("Source ID duplicated or crosses splits")
                    source_ids.add(source["id"])
                if split == "train" and count < report["splits"]["train"]["documents"]:
                    if document != next(base_train):
                        raise ValueError("Original training prefix changed")
                size = len(document["text"].encode("utf-8"))
                if (size != document["text_bytes"] or canonical_text(document["text"]) != document["text"] or
                        not settings["min_document_bytes"] <= size <= settings["max_document_bytes"]):
                    raise ValueError("Document byte count mismatch")
                ids = encode_document(document, tokenizer)
                packed = array("i", ids).tobytes()
                if binary.read(len(packed)) != packed:
                    raise ValueError("Packed IDs differ from frozen tokenizer")
                rebuilt.update(packed)
                if index != {"document_id": digest, "offset": tokens, "tokens": len(ids)}:
                    raise ValueError("Packed index mismatch")
                count += 1
                tokens += len(ids)
                text_bytes += size
                if count % 20000 == 0:
                    print(f"Verified {count:,} {split} documents", flush=True)
            if binary.read(1):
                raise ValueError("Extra packed tokens after documents")
        expected = manifest["splits"][split]
        if (count != expected["documents"] or tokens != expected["tokens"] or
                text_bytes != expected["text_bytes"] or tokens - count != expected["text_tokens"] or
                rebuilt.hexdigest() != expected["bin_sha256"] or
                sha256_file(output / f"{split}.index.jsonl") != expected["index_sha256"]):
            raise ValueError("Split statistics or packed checksums differ")
        statistics[split] = {"documents": count, "tokens_including_eos": tokens,
                             "text_tokens": tokens - count, "text_bytes": text_bytes}
    limits = manifest["limits"]
    train = statistics["train"]
    if (train["tokens_including_eos"] > limits["max_train_tokens"] or
            train["documents"] > limits["max_train_documents"] or
            train["text_bytes"] > limits["max_train_text_bytes"] or
            train["documents"] <= report["splits"]["train"]["documents"]):
        raise ValueError("Expansion capacity or growth check failed")
    if statistics["validation"] != {key: report["splits"]["validation"][key] for key in statistics["validation"]}:
        raise ValueError("Primary validation counts changed")
    verify_selection(output, base, manifest)
    return {"passed": True, "checkpoint": "CP4 preparation: corpus expansion, not training",
            "splits": statistics, "new_training_documents": train["documents"] - report["splits"]["train"]["documents"],
            "new_training_tokens": train["tokens_including_eos"] - report["splits"]["train"]["tokens_including_eos"],
            "manifest_sha256": sha256_file(output / "manifest.json"),
            "tokenizer_sha256": manifest["tokenizer_sha256"], "source_revision": manifest["source_revision"],
            "limits": limits, "selection_counts": manifest["selection_counts"],
            "files": manifest["files"],
            "checks": {"frozen_tokenizer_and_fit_provenance": True, "unchanged_primary_validation": True,
                       "original_training_prefix_preserved": True, "disjoint_content_and_source_ids": True,
                       "seeded_split_separation": True, "all_text_round_trips": True,
                       "source_audit_order_and_decisions_verified": True,
                       "every_packed_token_and_index_verified": True, "bounded_unique_training_growth": True},
            "full_batches_context512_batch16": (train["tokens_including_eos"] - 1) // 8192,
            "main_training_started": False, "limitations": manifest["limitations"]}
