"""Prepare a bounded corpus, fit training-only BPE, and verify saved artifacts."""
import argparse
import hashlib
import importlib.metadata
import itertools
import json
import os
import sys
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.request import urlopen
from zoneinfo import ZoneInfo

# A single tokenizer worker avoids CPU oversubscription and makes reruns comparable.
os.environ["RAYON_NUM_THREADS"] = "1"

import yaml
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from helixdepth.data import select_documents, split_for_hash, text_hash

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "validation")


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def write_rows(path: Path, rows: Iterator[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def prepare_documents(settings: dict[str, Any], output: Path) -> dict[str, Any]:
    from datasets import load_dataset

    if output.exists():
        raise FileExistsError(f"Preserving existing artifacts: {output}. Use --verify-only or --fit-existing.")
    source = load_dataset(settings["dataset_id"], name=settings["dataset_config"],
                          split=settings["source_split"], revision=settings["revision"], streaming=True)
    print("Reading bounded source prefix...", flush=True)

    def progress_rows() -> Iterator[dict[str, Any]]:
        for index, row in enumerate(itertools.islice(source, settings["candidate_documents"])):
            if index % 2000 == 0:
                print(f"Source documents: {index}", flush=True)
            yield row

    documents, selection_counts = select_documents(progress_rows(), settings)
    output.mkdir(parents=True)
    for split in SPLITS:
        write_rows(output / f"{split}.jsonl", (row for row in documents if row["split"] == split))
    card_url = (f"https://huggingface.co/datasets/{settings['dataset_id']}/raw/"
                f"{settings['revision']}/README.md")
    with urlopen(card_url, timeout=60) as response:
        (output / "source_card.md").write_bytes(response.read())
    manifest = {
        "settings": settings, "selection_counts": selection_counts,
        "source_card_url": card_url, "license": "ODC-By-1.0",
        "license_url": "https://opendatacommons.org/licenses/by/1-0/",
        "additional_terms": "https://commoncrawl.org/terms-of-use",
        "normalization": "Unicode NFC, CRLF/CR to LF, strip leading/trailing whitespace",
        "selection_rule": "First candidate_documents source rows; filter length/domains; group canonical text; sort by SHA256(seed:selection:text_sha256); greedily retain within document/UTF-8-byte caps",
        "split_rule": "int(SHA256(seed:split:text_sha256),16) % 1000 < validation_per_mille => validation",
        "benchmark_exclusion": "Wikipedia, Hugging Face and GitHub source domains excluded; no benchmark datasets loaded. Heuristic only: benchmark text elsewhere and near duplicates are not exhaustively checked.",
        "files": {f"{split}.jsonl": file_hash(output / f"{split}.jsonl") for split in SPLITS},
        "source_card_sha256": file_hash(output / "source_card.md"),
        "versions": {name: importlib.metadata.version(name) for name in ("datasets", "tokenizers", "PyYAML")},
        "python": sys.version, "tokenizer_workers": 1,
        "code_sha256": {"scripts/prepare_cp2.py": file_hash(Path(__file__)),
                        "src/helixdepth/data.py": file_hash(ROOT / "src/helixdepth/data.py")},
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def check_documents(output: Path, manifest: dict[str, Any]) -> dict[str, dict[str, int]]:
    settings = manifest["settings"]
    all_hashes: set[str] = set()
    all_ids: set[str] = set()
    statistics: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        path = output / f"{split}.jsonl"
        if file_hash(path) != manifest["files"][path.name]:
            raise ValueError(f"Split file checksum mismatch: {split}")
        statistics[split] = {"documents": 0, "source_documents": 0, "text_bytes": 0}
        for document in read_rows(path):
            digest = text_hash(document["text"])
            if digest != document["text_sha256"] or digest != document["document_id"]:
                raise ValueError("Document content hash mismatch")
            if digest in all_hashes:
                raise ValueError("Duplicate content appears more than once or across splits")
            if document["split"] != split or split_for_hash(digest, settings["seed"], settings["validation_per_mille"]) != split:
                raise ValueError("Incorrect document split assignment")
            all_hashes.add(digest)
            if not document["sources"]:
                raise ValueError("Document has no source IDs")
            for source in document["sources"]:
                if source["id"] in all_ids:
                    raise ValueError("Source document ID repeated or crosses splits")
                all_ids.add(source["id"])
                statistics[split]["source_documents"] += 1
            size = len(document["text"].encode("utf-8"))
            if size != document["text_bytes"] or not settings["min_document_bytes"] <= size <= settings["max_document_bytes"]:
                raise ValueError("Document byte size is invalid")
            statistics[split]["documents"] += 1
            statistics[split]["text_bytes"] += size
        if not statistics[split]["documents"]:
            raise ValueError(f"Empty split: {split}")
    if len(all_hashes) > settings["max_documents"] or sum(row["text_bytes"] for row in statistics.values()) > settings["max_text_bytes"]:
        raise ValueError("Corpus exceeds configured bound")
    if file_hash(output / "source_card.md") != manifest["source_card_sha256"]:
        raise ValueError("Source card checksum mismatch")
    return statistics


def fit_tokenizer(train_path: Path, settings: dict[str, Any]) -> tuple[Tokenizer, list[str]]:
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=settings["vocab_size"], min_frequency=settings["min_pair_frequency"],
                                  special_tokens=settings["special_tokens"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    fitted_ids: list[str] = []

    def training_texts() -> Iterator[str]:
        for row in read_rows(train_path):
            fitted_ids.append(row["document_id"])
            yield row["text"]

    tokenizer.train_from_iterator(training_texts(), trainer=trainer)
    if tokenizer.get_vocab_size(with_added_tokens=True) != settings["vocab_size"]:
        raise ValueError("Training corpus did not produce the exact requested vocabulary size")
    return tokenizer, fitted_ids


def encode_split(output: Path, split: str, tokenizer: Tokenizer) -> None:
    eos_id = tokenizer.token_to_id("<eos>")
    if eos_id is None:
        raise ValueError("Missing EOS token")

    def token_rows() -> Iterator[dict[str, Any]]:
        for document in read_rows(output / f"{split}.jsonl"):
            ids = tokenizer.encode(document["text"], add_special_tokens=False).ids
            yield {"document_id": document["document_id"], "input_ids": ids + [eos_id]}

    write_rows(output / f"{split}.tokens.jsonl", token_rows())


def verify(output: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    statistics = check_documents(output, manifest)
    settings = manifest["settings"]
    tokenizer = Tokenizer.from_file(str(output / "tokenizer.json"))
    vocab = tokenizer.get_vocab(with_added_tokens=True)
    if len(vocab) != 16384 or set(vocab.values()) != set(range(16384)):
        raise ValueError("Vocabulary must contain exactly 16,384 contiguous IDs")
    if not set(pre_tokenizers.ByteLevel.alphabet()).issubset(vocab):
        raise ValueError("Tokenizer is missing byte alphabet entries")
    special_ids = {token: tokenizer.token_to_id(token) for token in settings["special_tokens"]}
    if list(special_ids.values()) != list(range(len(special_ids))):
        raise ValueError("Special tokens have unexpected IDs")
    eos_id, unk_id = special_ids["<eos>"], special_ids["<unk>"]
    for split in SPLITS:
        statistics[split]["text_tokens"] = 0
        statistics[split]["tokens_including_eos"] = 0
        documents = read_rows(output / f"{split}.jsonl")
        token_rows = read_rows(output / f"{split}.tokens.jsonl")
        for document, token_row in itertools.zip_longest(documents, token_rows):
            if document is None or token_row is None or document["document_id"] != token_row["document_id"]:
                raise ValueError("Token/document rows do not match")
            expected = tokenizer.encode(document["text"], add_special_tokens=False).ids
            if token_row["input_ids"] != expected + [eos_id] or unk_id in expected:
                raise ValueError("Saved tokens do not match the shared tokenizer or contain UNK")
            if tokenizer.decode(expected, skip_special_tokens=False) != document["text"]:
                raise ValueError("Tokenizer text round trip failed")
            statistics[split]["text_tokens"] += len(expected)
            statistics[split]["tokens_including_eos"] += len(expected) + 1
    for probe in ("English text\nwith spacing.", "café 中文 🧬", "\t leading and trailing "):
        encoded = tokenizer.encode(probe, add_special_tokens=False)
        if tokenizer.decode(encoded.ids, skip_special_tokens=False) != probe or unk_id in encoded.ids:
            raise ValueError("Byte-level Unicode round trip failed")
    artifact_names = ("manifest.json", "source_card.md", "train.jsonl", "validation.jsonl", "tokenizer.json",
                      "train.tokens.jsonl", "validation.tokens.jsonl", "tokenizer_training.json")
    provenance = json.loads((output / "tokenizer_training.json").read_text(encoding="utf-8"))
    if provenance["training_file_sha256"] != manifest["files"]["train.jsonl"] or provenance["training_split"] != "train":
        raise ValueError("Tokenizer training provenance mismatch")
    expected_fit_ids = [row["document_id"] for row in read_rows(output / "train.jsonl")]
    if provenance["fitted_document_ids"] != expected_fit_ids:
        raise ValueError("Tokenizer did not consume exactly the saved training documents")
    return {"checkpoint": "CP2", "passed": True, "verified_at": datetime.now(ZoneInfo("Asia/Manila")).isoformat(),
            "settings": settings, "selection_counts": manifest["selection_counts"], "splits": statistics,
            "vocab_size_including_special_tokens": len(vocab), "special_token_ids": special_ids,
            "checks": {"document_hashes": True, "disjoint_content_and_source_ids": True,
                       "seeded_split_assignments": True, "bounded_corpus": True, "training_only_fit_provenance": True,
                       "saved_tokens_match_shared_tokenizer": True, "all_document_round_trips": True,
                       "unicode_byte_fallback": True},
            "files": {name: {"sha256": file_hash(output / name), "bytes": (output / name).stat().st_size}
                      for name in artifact_names},
            "limitations": ["Bounded source prefix is not a uniform sample of the whole dataset",
                            "Exact/canonical duplicate grouping; near-duplicate detection is not implemented",
                            manifest["benchmark_exclusion"], "No language-model training or quality evaluation performed"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/data.yaml")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--fit-existing", action="store_true")
    parser.add_argument("--report", type=Path, default=ROOT / "docs/results/cp2.json")
    args = parser.parse_args()
    if args.verify_only and args.fit_existing:
        parser.error("Choose only one existing-artifact mode")
    settings = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = ROOT / settings["output_dir"]
    if args.verify_only or args.fit_existing:
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if settings != manifest["settings"]:
            raise ValueError("Current config differs from saved corpus settings")
    else:
        manifest = prepare_documents(settings, output)
    if not args.verify_only:
        check_documents(output, manifest)
        print("Fitting BPE using training documents only...", flush=True)
        tokenizer, fitted_ids = fit_tokenizer(output / "train.jsonl", settings)
        tokenizer.save(str(output / "tokenizer.json"))
        write_json(output / "tokenizer_training.json", {
            "training_split": "train", "training_file_sha256": manifest["files"]["train.jsonl"],
            "fitted_document_ids": fitted_ids,
            "fitting_code_sha256": file_hash(Path(__file__)),
            "vocab_size": settings["vocab_size"], "special_tokens": settings["special_tokens"],
            "initial_alphabet_size": 256, "min_frequency": settings["min_pair_frequency"],
            "add_prefix_space": False, "tokenizer_workers": 1,
            "normalizer": None, "post_processor": None, "eos_policy": "Append one EOS ID per document when saving token rows"})
        for split in SPLITS:
            print(f"Encoding {split}...", flush=True)
            encode_split(output, split, tokenizer)
    print("Verifying every saved document and token row...", flush=True)
    report = verify(output, manifest)
    if args.verify_only and args.report.exists():
        previous = json.loads(args.report.read_text(encoding="utf-8"))
        if report["files"] != previous["files"]:
            raise ValueError("Artifacts differ from previous verified report")
    write_json(args.report, report)
    print(json.dumps({"passed": report["passed"], "vocab_size": report["vocab_size_including_special_tokens"],
                      "splits": report["splits"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
