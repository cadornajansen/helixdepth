"""Discarded four-update GPU mechanics check, never a candidate model."""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from helixdepth.config import ModelConfig
from helixdepth.packing import sha256_file
from helixdepth.training import Trainer, TrainingConfig
from scripts.run_continuation import initialize, atomic_json
from scripts.train import assert_identical


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--child", choices=("full", "interrupted", "resumed"))
    args = parser.parse_args()
    base = Path("artifacts/corpus_100m_v1")
    parent_path = Path("artifacts/main_helixdepth/latest.pt")
    data = args.output / "fixture_data"
    if args.child:
        parent = torch.load(parent_path, map_location="cpu", weights_only=True)
        config = TrainingConfig(context=512, batch_size=16, learning_rate=1e-4,
                                schedule_steps=4, warmup_steps=1)
        trainer = Trainer(ModelConfig(**parent["model_config"]), config, data, "cuda")
        if args.child == "resumed":
            trainer.load_checkpoint(args.output / "interrupted.pt")
        else:
            initialize(trainer, parent)
        for _ in range((2 if args.child == "interrupted" else 4) - trainer.step):
            print(trainer.train_step(), flush=True)
        trainer.save_checkpoint(args.output / f"{args.child}.pt")
        return
    if args.output.exists():
        raise FileExistsError(args.output)
    data.mkdir(parents=True)
    manifest = json.loads((base / "manifest.json").read_text())
    # Reused text is strictly for a discarded mechanics test, not additional training.
    with (base / "train.bin").open("rb") as stream:
        (data / "train.bin").write_bytes(stream.read((8192 * 4 + 1) * 4))
    shutil.copyfile(base / "validation.bin", data / "validation.bin")
    manifest["splits"]["train"] = {"tokens": 8192 * 4 + 1, "bin_sha256": sha256_file(data / "train.bin")}
    atomic_json(data / "manifest.json", manifest)
    parent_hash = sha256_file(parent_path)
    for child in ("full", "interrupted", "resumed"):
        subprocess.run([sys.executable, "-m", "scripts.check_continuation_gpu", "--output", str(args.output),
                        "--child", child], check=True, timeout=180)
    assert_identical(torch.load(args.output / "full.pt", map_location="cpu", weights_only=True),
                     torch.load(args.output / "resumed.pt", map_location="cpu", weights_only=True))
    assert sha256_file(parent_path) == parent_hash
    atomic_json(args.output / "result.json", {"passed": True, "device": "cuda", "fresh_process_exact_resume": True,
                "parent_sha256": parent_hash, "scope": "Discarded 4-update mechanics fixture; no capability claim"})
    print("GPU CONTINUATION RECOVERY PASSED", flush=True)


if __name__ == "__main__":
    main()
