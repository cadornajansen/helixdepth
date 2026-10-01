"""Freeze a completed checkpoint and its recovery metadata for downloading."""
import argparse
import json
import shutil
import tarfile
from pathlib import Path
from typing import Any

import torch

from verify_transfer import sha256, verify_files

ROOT = Path(__file__).resolve().parents[1]


def backup(checkpoint: Path, destination: Path) -> dict[str, Any]:
    if checkpoint.suffix != ".pt" or not checkpoint.is_file():
        raise ValueError("Use a completed .pt checkpoint, never a .tmp file")
    if destination.exists():
        raise FileExistsError(f"Preserving backup: {destination}")
    archive = destination.with_suffix(".tar.gz")
    if archive.exists() or Path(str(archive) + ".sha256").exists():
        raise FileExistsError(f"Preserving previous archive: {archive}")
    destination.mkdir(parents=True)
    # Atomic rename in the trainer means this copy opens either complete version.
    shutil.copyfile(checkpoint, destination / "latest.pt")
    saved = torch.load(destination / "latest.pt", map_location="cpu", weights_only=True)
    for name, value in (("training_used.json", saved["training_config"]),
                        ("model_used.json", saved["model_config"]),
                        ("recovery.json", {"step": saved["step"], "runtime": saved["runtime"],
                                           "data": saved["data"], "validation_identity": saved["validation_identity"],
                                           "tokens_processed": saved["tokens_processed"]})):
        (destination / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    shutil.copyfile(ROOT / "requirements/runpod.lock", destination / "runpod.lock")
    shutil.copyfile(ROOT / "artifacts/cp3_data/manifest.json", destination / "data_manifest.json")
    if (ROOT / "transfer_manifest.json").exists():
        shutil.copyfile(ROOT / "transfer_manifest.json", destination / "original_transfer_manifest.json")
    manifest = {"files": {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
                          for path in destination.iterdir() if path.is_file()}}
    (destination / "checkpoint_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    verify_files(destination, manifest)
    with tarfile.open(archive, "w:gz") as output:
        for path in destination.iterdir():
            output.add(path, arcname=path.name, recursive=False)
    checksum = sha256(archive)
    Path(str(archive) + ".sha256").write_text(f"{checksum}  {archive.name}\n", encoding="utf-8")
    return {"archive": str(archive), "archive_sha256": checksum, "step": saved["step"],
            "tokens_processed": saved["tokens_processed"], "files_verified": len(manifest["files"])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(backup(args.checkpoint, args.output), indent=2))


if __name__ == "__main__":
    main()
