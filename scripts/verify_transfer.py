"""Standard-library checksum verification; runs before installing dependencies."""
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_files(root: Path, manifest: dict[str, Any]) -> int:
    root = root.resolve()
    for name, expected in manifest["files"].items():
        path = root / name
        if Path(name).is_absolute() or ".." in Path(name).parts or not path.resolve().is_relative_to(root):
            raise ValueError(f"Unsafe manifest path: {name}")
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing or non-regular file: {name}")
        if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"Checksum mismatch: {name}")
    return len(manifest["files"])


def verify_package(root: Path) -> dict[str, Any]:
    package = json.loads((root / "transfer_manifest.json").read_text(encoding="utf-8"))
    count = verify_files(root, package)
    directory = package.get("data_directory", "artifacts/cp3_data")
    if directory not in ("artifacts/cp3_data", "artifacts/corpus_100m_v1"):
        raise ValueError("Unrecognized packed corpus")
    packed = json.loads((root / directory / "manifest.json").read_text(encoding="utf-8"))
    expanded = directory == "artifacts/corpus_100m_v1"
    if expanded:
        expansion = json.loads((root / "docs/results/corpus_expansion.json").read_text(encoding="utf-8"))
        if not expansion["passed"] or sha256(root / directory / "manifest.json") != expansion["manifest_sha256"]:
            raise ValueError("Expanded corpus verification evidence mismatch")
    tokenizer_path = root / "artifacts/cp2/tokenizer.json"
    tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    vocabulary = set(tokenizer["model"]["vocab"].values()) | {row["id"] for row in tokenizer["added_tokens"]}
    if vocabulary != set(range(16384)) or sha256(tokenizer_path) != packed["tokenizer_sha256"]:
        raise ValueError("Tokenizer vocabulary/hash mismatch")
    ids: dict[str, set[str]] = {}
    for split in ("train", "validation"):
        info = packed["splits"][split]
        binary = root / directory / f"{split}.bin"
        index = root / directory / f"{split}.index.jsonl"
        if (sha256(binary) != info["bin_sha256"] or binary.stat().st_size != info["tokens"] * 4
                or sha256(index) != info["index_sha256"]):
            raise ValueError(f"Packed manifest mismatch: {split}")
        offset = 0
        ids[split] = set()
        for line in index.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["offset"] != offset or row["tokens"] < 1 or row["document_id"] in ids[split]:
                raise ValueError("Invalid document index")
            ids[split].add(row["document_id"])
            offset += row["tokens"]
        if offset != info["tokens"] or len(ids[split]) != info["documents"]:
            raise ValueError("Packed document counts mismatch")
    if ids["train"] & ids["validation"]:
        raise ValueError("Training/validation document overlap")
    fitting = json.loads((root / "artifacts/cp2/tokenizer_training.json").read_text(encoding="utf-8"))
    fitted_ids = set(fitting["fitted_document_ids"])
    fitting_matches = fitted_ids <= ids["train"] if expanded else fitted_ids == ids["train"]
    if fitting["training_split"] != "train" or not fitting_matches or fitted_ids & ids["validation"]:
        raise ValueError("Tokenizer fitting IDs differ from training documents")
    cp2 = json.loads((root / "artifacts/cp2/manifest.json").read_text(encoding="utf-8"))
    if cp2["settings"]["revision"] != packed["source_revision"]:
        raise ValueError("Source revision mismatch")
    evidence = json.loads((root / "docs/results/cp2.json").read_text(encoding="utf-8"))
    if not evidence["passed"]:
        raise ValueError("CP2 verification evidence did not pass")
    for name in ("tokenizer.json", "tokenizer_training.json", "manifest.json"):
        if sha256(root / "artifacts/cp2" / name) != evidence["files"][name]["sha256"]:
            raise ValueError(f"CP2 provenance checksum mismatch: {name}")
    return {"passed": True, "files_verified": count, "vocabulary_entries": len(vocabulary),
            "split_overlap": 0, "tokenizer_fitting_ids_are_training_only": True,
            "data_directory": directory, "fitted_documents": len(fitted_ids),
            "splits": packed["splits"], "source_revision": packed["source_revision"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path, help="Verify a checkpoint backup file manifest instead")
    args = parser.parse_args()
    result = ({"passed": True, "files_verified": verify_files(args.root, json.loads(args.manifest.read_text(encoding="utf-8")))}
              if args.manifest else verify_package(args.root))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
