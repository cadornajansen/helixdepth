"""Explicit weights-only new stage; original exact-resume rules stay unchanged."""
import argparse
import json
import math
import os
import signal
import shutil
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from helixdepth.config import ModelConfig
from helixdepth.packing import sha256_file
from helixdepth.training import Trainer, TrainingConfig


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def initialize(trainer: Trainer, parent: dict[str, Any]) -> None:
    if parent["model_config"] != trainer.model.config.to_dict():
        raise ValueError("Parent architecture mismatch")
    if parent["validation_identity"] != trainer.validation_data.identity:
        raise ValueError("Frozen validation mismatch")
    if parent["data"]["identity"]["tokenizer_sha256"] != trainer.train_data.identity["tokenizer_sha256"]:
        raise ValueError("Tokenizer mismatch")
    if parent["data"]["identity"]["bin_sha256"] == trainer.train_data.identity["bin_sha256"]:
        raise ValueError("New stage must use fresh training data")
    trainer.model.load_state_dict(parent["model"], strict=True)


def save_candidate(trainer: Trainer, output: Path, loss: float) -> None:
    temporary = output / "best.pt.tmp"
    with temporary.open("wb") as stream:
        torch.save({"config": trainer.model.config.to_dict(), "state_dict": trainer.model.state_dict(),
                    "selection": {"stage_step": trainer.step, "full_validation_loss": loss}}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output / "best.pt")


def run(parent_path: Path, data: Path, output: Path, device: str, deadline: float,
        resume: bool = False, stop_after: int | None = None, parent_report: Path | None = None) -> None:
    started = time.time()
    if started + 120 >= deadline:
        raise ValueError("Not enough time before deadline")
    manifest = json.loads((data / "manifest.json").read_text())
    if not manifest.get("fresh_verified"):
        raise ValueError("Fresh corpus verification required")
    for name, info in manifest["files"].items():
        if sha256_file(data / name) != info["sha256"]:
            raise ValueError(f"Fresh corpus changed: {name}")
    steps = (manifest["splits"]["train"]["tokens"] - 1) // (16 * 512)
    config = TrainingConfig(context=512, batch_size=16, learning_rate=1e-4,
                            schedule_steps=steps, warmup_steps=min(244, steps // 10),
                            checkpoint_every=500, validation_batches=64)
    parent = torch.load(parent_path, map_location="cpu", weights_only=True)
    parent_targets = parent["tokens_processed"]
    if parent_report is not None:
        prior = json.loads(parent_report.read_text())
        if (prior["files"]["latest.pt"]["sha256"] != sha256_file(parent_path)
                or prior["stage_targets"] != parent_targets or prior["stop_reason"] != "schedule_complete"):
            raise ValueError("Parent stage report mismatch")
        parent_targets = prior["cumulative_targets"]
    trainer = Trainer(ModelConfig(**parent["model_config"]), config, data, device)
    identity = {"parent_sha256": sha256_file(parent_path), "parent_targets": parent_targets,
                "parent_report_sha256": sha256_file(parent_report) if parent_report else None,
                "data_manifest_sha256": sha256_file(data / "manifest.json"), "config": asdict(config),
                "deadline_unix": deadline, "optimizer": "New AdamW state; weights-only stage initialization",
                "runner_sha256": sha256_file(Path(__file__)),
                "training_sha256": sha256_file(Path(__file__).resolve().parents[1] / "src/helixdepth/training.py")}
    if resume:
        if json.loads((output / "stage.json").read_text()) != identity:
            raise ValueError("Stage identity changed")
        trainer.load_checkpoint(output / "latest.pt")
        best = torch.load(output / "best.pt", map_location="cpu", weights_only=True)["selection"]
    else:
        if output.exists():
            raise FileExistsError(output)
        initialize(trainer, parent)
        output.mkdir(parents=True)
        atomic_json(output / "stage.json", identity)
        initial = trainer.validate()
        if not math.isfinite(initial["loss"]):
            raise ValueError("Invalid initial validation")
        best = {"stage_step": 0, "full_validation_loss": initial["loss"]}
        save_candidate(trainer, output, initial["loss"])
        trainer.save_checkpoint(output / "latest.pt")
        print(f"Initial full validation: {initial}", flush=True)
    del parent
    stopping = False

    def stop(signum: int, frame: Any) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    limit = min(steps, stop_after) if stop_after else steps
    if limit <= trainer.step:
        raise ValueError("Stopping step must advance")
    reason = "schedule_complete" if limit == steps else "requested_stop"
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with (output / "progress.jsonl").open("a", encoding="utf-8", buffering=1) as progress:
        while trainer.step < limit:
            if stopping or time.time() >= deadline - 120:
                reason = "deadline_or_signal"
                break
            metric = trainer.train_step()
            if not math.isfinite(float(metric["loss"])):
                raise FloatingPointError("Non-finite training loss; previous checkpoint preserved")
            progress.write(json.dumps(metric) + "\n")
            if trainer.step % 100 == 0:
                print(f"additional step {trainer.step}/{steps}: loss={metric['loss']:.4f}", flush=True)
            if trainer.step % 500 == 0:
                trainer.save_checkpoint(output / "latest.pt")
            if trainer.step % 2000 == 0 or trainer.step == limit:
                if time.time() >= deadline - 120:
                    reason = "deadline_before_validation"
                    break
                validation = trainer.validate()
                if not math.isfinite(validation["loss"]):
                    raise FloatingPointError("Non-finite validation")
                progress.write(json.dumps({"validation": validation, "step": trainer.step}) + "\n")
                print(f"Full validation step {trainer.step}: {validation}", flush=True)
                if validation["loss"] < best["full_validation_loss"]:
                    best = {"stage_step": trainer.step, "full_validation_loss": validation["loss"]}
                    save_candidate(trainer, output, validation["loss"])
    trainer.save_checkpoint(output / "latest.pt")
    shutil.copyfile(output / "best.pt", output / "model.pt.tmp")
    os.replace(output / "model.pt.tmp", output / "model.pt")
    atomic_json(output / "metrics.json", {"stage": f"{trainer.model.config.variant} additional training", "stop_reason": reason,
                "step": trainer.step, "schedule_steps": steps, "stage_targets": trainer.train_data.offset,
                "cumulative_targets": identity["parent_targets"] + trainer.train_data.offset,
                "selected": best, "wall_seconds_this_process": time.time() - started,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
                "files": {name: {"sha256": sha256_file(output / name), "bytes": (output / name).stat().st_size}
                          for name in ("model.pt", "latest.pt", "best.pt")}, "identity": identity})
    print(f"STAGE FINISHED: {best}; {reason}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initialize-from", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--deadline", required=True, help="ISO-8601 time including timezone")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--parent-report", type=Path, help="Verified completed stage report for cumulative accounting")
    args = parser.parse_args()
    deadline = datetime.fromisoformat(args.deadline)
    if deadline.tzinfo is None:
        parser.error("Deadline requires timezone")
    run(args.initialize_from, args.data, args.output, args.device, deadline.timestamp(), args.resume, args.stop_after,
        args.parent_report)


if __name__ == "__main__":
    main()
