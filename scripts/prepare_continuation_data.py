"""Build a fresh, single-pass stage without prepending the original corpus."""
import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

os.environ.setdefault("RAYON_NUM_THREADS", "1")

from tokenizers import Tokenizer
from helixdepth.expansion import new_documents, read_rows, write_json, write_split
from helixdepth.packing import sha256_file


def build(source: Iterable[dict[str, Any]], base: Path, output: Path,
          max_tokens: int = 100_000_000, candidates: int = 400_000,
          minimum_tokens: int = 1_000_000, previous: Path | None = None) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((base / "manifest.json").read_text())
    for name in ("train.jsonl", "validation.jsonl", "selection_audit.jsonl",
                 "tokenizer.json", "validation.bin", "validation.index.jsonl"):
        if sha256_file(base / name) != manifest["files"][name]["sha256"]:
            raise ValueError(f"Original identity mismatch: {name}")
    tokenizer = Tokenizer.from_file(str(base / "tokenizer.json"))
    old_ids = {row["source"]["id"] for row in read_rows(base / "selection_audit.jsonl")}
    old_hashes = {row["text_sha256"] for row in read_rows(base / "selection_audit.jsonl")}
    source_start = manifest["selection_counts"]["scanned_documents"]
    if previous is not None:
        prior = json.loads((previous / "manifest.json").read_text())
        if prior["source_settings"] != manifest["source_settings"] or not prior.get("fresh_verified"):
            raise ValueError("Previous fresh corpus identity mismatch")
        for name in ("train.jsonl", "selection_audit.jsonl"):
            if sha256_file(previous / name) != prior["files"][name]["sha256"]:
                raise ValueError(f"Previous corpus changed: {name}")
        for row in read_rows(previous / "selection_audit.jsonl"):
            old_ids.add(row["source"]["id"])
            old_hashes.add(row["text_sha256"])
        old_hashes.update(row["text_sha256"] for row in read_rows(previous / "train.jsonl"))
        source_start = max(source_start, prior["source_start_index"] + prior["selection_counts"]["scanned_documents"])
    output.mkdir(parents=True)
    counts: Counter[str] = Counter()
    with (output / "selection_audit.jsonl").open("w", encoding="utf-8") as audit:
        documents = new_documents(source, base, manifest["source_settings"], candidates, audit, counts)

        def fresh_documents() -> Iterator[dict[str, Any]]:
            for doc in documents:
                if doc["text_sha256"] in old_hashes or any(s["id"] in old_ids for s in doc["sources"]):
                    counts["excluded_prior_audit"] += 1
                    continue
                yield doc

        train = write_split(fresh_documents(), output, "train", tokenizer,
                            {"max_train_tokens": max_tokens, "max_train_documents": 200_000,
                             "max_train_text_bytes": 600_000_000})
    if train["tokens"] < minimum_tokens:
        raise ValueError("Insufficient fresh training text")
    for name in ("tokenizer.json", "validation.bin", "validation.index.jsonl", "validation.jsonl", "source_card.md"):
        shutil.copyfile(base / name, output / name)
    # Independently re-encode saved rows and compare every packed byte and index.
    old_accepted = {row["text_sha256"] for split in ("train", "validation")
                    for row in read_rows(base / f"{split}.jsonl")}
    from array import array
    from helixdepth.expansion import encode_document
    offset = 0
    seen: set[str] = set()
    with (output / "train.bin").open("rb") as binary:
        indexes = iter(read_rows(output / "train.index.jsonl"))
        for doc in read_rows(output / "train.jsonl"):
            digest = doc["text_sha256"]
            if digest in old_accepted or digest in old_hashes or digest in seen:
                raise ValueError("Fresh data overlap")
            seen.add(digest)
            ids = encode_document(doc, tokenizer)
            expected = array("i", ids).tobytes()
            if binary.read(len(expected)) != expected:
                raise ValueError("Token verification failed")
            index = next(indexes)
            if index != {"document_id": doc["document_id"], "offset": offset, "tokens": len(ids)}:
                raise ValueError("Index mismatch")
            offset += len(ids)
        if binary.read(1) or next(indexes, None) is not None or offset != train["tokens"]:
            raise ValueError("Unexpected tail")
    result = {"format": "little-endian-int32", "vocab_size": 16384, "eos_id": 2,
              "tokenizer_sha256": manifest["tokenizer_sha256"],
              "source_revision": manifest["source_revision"], "source_settings": manifest["source_settings"],
              "parent_manifest_sha256": sha256_file(base / "manifest.json"),
              "source_start_index": source_start,
              "previous_manifest_sha256": sha256_file(previous / "manifest.json") if previous else None,
              "audit_index_note": "source_index is relative to source_start_index",
              "splits": {"train": train, "validation": manifest["splits"]["validation"]},
              "selection_counts": dict(counts), "fresh_verified": True,
              "limitations": ["Exact canonical hashes and source IDs excluded across old corpus and audit; near duplicates not exhaustively excluded", "Domain exclusions do not prove benchmark decontamination"],
              "files": {p.name: {"sha256": sha256_file(p), "bytes": p.stat().st_size}
                        for p in output.iterdir() if p.is_file()}}
    write_json(output / "manifest.json", result)
    print(json.dumps({"fresh_verified": True, "train": train}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path("artifacts/corpus_100m_v1"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous", type=Path, help="Additional prior fresh corpus to exclude")
    args = parser.parse_args()
    manifest = json.loads((args.base / "manifest.json").read_text())
    settings = manifest["source_settings"]
    from datasets import load_dataset
    source = load_dataset(settings["dataset_id"], name=settings["dataset_config"],
                          revision=settings["revision"], split=settings["source_split"], streaming=True)
    source_start = manifest["selection_counts"]["scanned_documents"]
    if args.previous:
        prior = json.loads((args.previous / "manifest.json").read_text())
        source_start = max(source_start, prior["source_start_index"] + prior["selection_counts"]["scanned_documents"])
    source = source.skip(source_start)
    build(source, args.base, args.output, previous=args.previous)


if __name__ == "__main__":
    main()
