"""Verify both downloaded continuation stages before another training run."""
import argparse
import hashlib
import json
from pathlib import Path


def verify(root: Path) -> dict[str, object]:
    models: dict[str, dict[str, str]] = {}
    for variant in ("baseline", "helixdepth"):
        directory = root / f"{variant}_additional_100m_v1"
        report = json.loads((directory / "metrics.json").read_text())
        if report["stop_reason"] != "schedule_complete":
            raise ValueError(f"Incomplete stage: {variant}")
        hashes: dict[str, str] = {}
        for name, identity in report["files"].items():
            path = directory / name
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != identity["sha256"] or path.stat().st_size != identity["bytes"]:
                raise ValueError(f"Backup mismatch: {variant}/{name}")
            hashes[name] = digest
        models[variant] = hashes
        print(f"{variant}: BACKUP VERIFIED", flush=True)
    result: dict[str, object] = {"verified": True, "models": models}
    (root / "continuation_200m_backup_verified.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    verify(args.root)


if __name__ == "__main__":
    main()
