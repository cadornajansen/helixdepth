"""Deterministic document selection and splitting, independent of tokenization."""
import hashlib
import unicodedata
from collections import Counter
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlsplit


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_text(text: str) -> str:
    """Group exact duplicates and duplicates differing only in these conventions."""
    return unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n")).strip()


def seeded_hash(seed: int, purpose: str, content_hash: str) -> str:
    return text_hash(f"{seed}:{purpose}:{content_hash}")


def split_for_hash(content_hash: str, seed: int, validation_per_mille: int) -> str:
    bucket = int(seeded_hash(seed, "split", content_hash), 16) % 1000
    return "validation" if bucket < validation_per_mille else "train"


def select_documents(rows: Iterable[dict[str, Any]], settings: dict[str, Any]
                     ) -> tuple[list[dict[str, Any]], dict[str, int]]:
    groups: dict[str, dict[str, Any]] = {}
    seen_ids: dict[str, str] = {}
    counts: Counter[str] = Counter()
    for index, row in enumerate(rows):
        if index >= settings["candidate_documents"]:
            break
        counts["scanned_documents"] += 1
        raw = row[settings["text_column"]]
        if not isinstance(raw, str):
            raise ValueError("Source text must be a string")
        document = canonical_text(raw)
        size = len(document.encode("utf-8"))
        if not settings["min_document_bytes"] <= size <= settings["max_document_bytes"]:
            counts["excluded_length"] += 1
            continue
        host = (urlsplit(str(row.get("url", ""))).hostname or "").lower()
        if any(host == domain or host.endswith("." + domain)
               for domain in settings["excluded_domains"]):
            counts["excluded_domain"] += 1
            continue
        digest = text_hash(document)
        source_id = str(row["id"])
        if source_id in seen_ids:
            if seen_ids[source_id] != digest:
                raise ValueError("One source ID has conflicting content")
            counts["repeated_source_ids"] += 1
            continue
        seen_ids[source_id] = digest
        source = {"id": source_id, "url": row.get("url"), "dump": row.get("dump"),
                  "raw_text_sha256": text_hash(raw), "source_index": index}
        if digest not in groups:
            groups[digest] = {"document_id": digest, "text_sha256": digest,
                              "text": document, "text_bytes": size, "sources": []}
        else:
            counts["duplicate_documents"] += 1
        groups[digest]["sources"].append(source)
    counts["eligible_unique_documents"] = len(groups)
    ordered = sorted(groups.values(), key=lambda item: (
        seeded_hash(settings["seed"], "selection", item["text_sha256"]), item["text_sha256"]))
    selected: list[dict[str, Any]] = []
    used_bytes = 0
    for document in ordered:
        if len(selected) >= settings["max_documents"]:
            break
        if used_bytes + document["text_bytes"] > settings["max_text_bytes"]:
            counts["skipped_byte_budget"] += 1
            continue
        document["split"] = split_for_hash(document["text_sha256"], settings["seed"],
                                            settings["validation_per_mille"])
        document["sources"].sort(key=lambda source: source["id"])
        selected.append(document)
        used_bytes += document["text_bytes"]
    counts["selected_unique_documents"] = len(selected)
    counts["selected_text_bytes"] = used_bytes
    if not selected or {item["split"] for item in selected} != {"train", "validation"}:
        raise ValueError("Selection must contain both training and validation documents")
    return selected, dict(counts)
