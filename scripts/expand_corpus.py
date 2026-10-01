"""Prepare and verify a larger local corpus with the frozen CP2 tokenizer."""
import argparse
import json
import os
from pathlib import Path

os.environ["RAYON_NUM_THREADS"] = "1"

import yaml

from helixdepth.expansion import checked_base, expand_corpus, verify_expansion, write_json

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/expansion.yaml")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path, default=ROOT / "docs/results/corpus_expansion.json")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    base, evidence = ROOT / config["base_dir"], ROOT / config["base_evidence"]
    output = args.output or ROOT / config["output_dir"]
    limits = {key: value for key, value in config.items() if key not in ("base_dir", "base_evidence", "output_dir")}
    if not args.verify_only:
        if output.exists():
            raise FileExistsError(f"Preserving existing corpus: {output}; use --verify-only")
        from datasets import load_dataset

        settings = checked_base(base, evidence)["settings"]
        source = load_dataset(settings["dataset_id"], name=settings["dataset_config"],
                              revision=settings["revision"], split=settings["source_split"], streaming=True)
        expand_corpus(source, base, evidence, output, limits)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if manifest["limits"] != limits:
        raise ValueError("Saved expansion limits differ from config")
    report = verify_expansion(output, base, evidence)
    if args.verify_only and args.report.exists():
        previous = json.loads(args.report.read_text(encoding="utf-8"))
        if previous["manifest_sha256"] != report["manifest_sha256"]:
            raise ValueError("Expanded manifest changed since previous verified report")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.report, report)
    print(json.dumps({key: report[key] for key in ("passed", "splits", "new_training_documents")}, indent=2))


if __name__ == "__main__":
    main()
