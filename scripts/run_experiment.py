"""Matched single-pass runs, with a separate fresh-process recovery gate."""
import argparse
import gc
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from helixdepth.config import ModelConfig
from helixdepth.training import Trainer, TrainingConfig
try:
    from .train import assert_identical, write_json
    from .verify_transfer import sha256
except ImportError:  # Direct script execution on the Pod.
    from train import assert_identical, write_json
    from verify_transfer import sha256

ROOT = Path(__file__).resolve().parents[1]


def train_run(variant: str, config: TrainingConfig, data: Path, output: Path,
              stop_after: int, resume: Path | None, device: str,
              full_validation: bool = True, max_seconds: float = 10800) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Preserving existing run: {output}")
    started = time.monotonic()
    trainer = Trainer(ModelConfig.from_yaml(ROOT / f"configs/{variant}.yaml"), config, data, device)
    if resume:
        trainer.load_checkpoint(resume)
    if not trainer.step < stop_after <= config.schedule_steps:
        raise ValueError("Stopping step must advance within the original schedule")
    output.mkdir(parents=True)
    initial = trainer.validate(config.validation_batches) if full_validation else None
    validations: list[dict[str, Any]] = []
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    step_seconds = 0.0
    initial_step = trainer.step
    with (output / "progress.jsonl").open("w", encoding="utf-8", buffering=1) as log:
        while trainer.step < stop_after:
            if time.monotonic() - started > max_seconds:
                trainer.save_checkpoint(output / "latest.pt")
                raise TimeoutError("Run time cap reached; latest complete step saved. Resume into a new output directory.")
            if device == "cuda":
                torch.cuda.synchronize()
            before = time.monotonic()
            metric = trainer.train_step()
            if device == "cuda":
                torch.cuda.synchronize()
            step_seconds += time.monotonic() - before
            log.write(json.dumps(metric) + "\n")
            if trainer.step % 100 == 0 or trainer.step == stop_after:
                print(f"{variant} step {trainer.step}/{stop_after}: loss={metric['loss']:.4f}; "
                      f"training targets/sec={(trainer.step-initial_step)*config.batch_size*config.context/step_seconds:.0f}", flush=True)
            if full_validation and trainer.step % 1000 == 0:
                validation = {"step": trainer.step, **trainer.validate(config.validation_batches)}
                validations.append(validation)
                write_json(output / "validation_progress.json", validations)
                print(f"validation: {validation}", flush=True)
            if trainer.step % config.checkpoint_every == 0:
                trainer.save_checkpoint(output / "latest.pt")
    final = trainer.validate() if full_validation else None
    trainer.save_checkpoint(output / "latest.pt")
    trainer.model.save(output / "model.pt")
    report = {
        "variant": variant, "complete_schedule": trainer.step == config.schedule_steps,
        "step": trainer.step, "target_tokens": trainer.train_data.offset,
        "config": asdict(config), "runtime": trainer.runtime_identity(),
        "initial_validation_prefix": initial, "periodic_validation_prefix": validations,
        "final_validation_full": final, "training_step_seconds_this_process": step_seconds,
        "wall_seconds_this_process": time.monotonic() - started,
        "resumed_from": str(resume) if resume else None,
        "data": trainer.train_data.state_dict(), "validation_identity": trainer.validation_data.identity,
        "model_parameters": sum(p.numel() for p in trainer.model.parameters()),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved() if device == "cuda" else None,
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
        "files": {name: {"sha256": sha256(output / name), "bytes": (output / name).stat().st_size}
                  for name in ("latest.pt", "model.pt", "progress.jsonl")},
        "scope": "One matched corpus pass; validation is FineWeb-Edu, not a benchmark accuracy score",
    }
    write_json(output / "metrics.json", report)
    return report


def recovery(config_path: Path, data: Path, output: Path, device: str) -> None:
    if output.exists():
        raise FileExistsError(f"Preserving previous recovery check: {output}")
    config = TrainingConfig.from_yaml(config_path)
    if not config.deterministic:
        raise ValueError("Exact recovery requires deterministic settings")
    results = []
    for variant in ("baseline", "helixdepth"):
        runs = {}
        for name, steps, previous in (("full", 4, None), ("interrupted", 2, None), ("resumed", 4, "interrupted")):
            destination = output / variant / name
            command = [sys.executable, str(Path(__file__).resolve()), "--mode", "run", "--variant", variant,
                       "--config", str(config_path), "--data", str(data), "--output", str(destination),
                       "--device", device, "--stop-after", str(steps), "--recovery-child"]
            if previous:
                command += ["--resume", str(output / variant / previous / "latest.pt")]
            subprocess.run(command, check=True, timeout=1800)
            runs[name] = json.loads((destination / "metrics.json").read_text())
        full = torch.load(output / variant / "full/latest.pt", map_location="cpu", weights_only=True)
        resumed = torch.load(output / variant / "resumed/latest.pt", map_location="cpu", weights_only=True)
        assert_identical(full, resumed)
        results.append({"variant": variant, "exact_match": True, "fresh_process": True, "runs": runs,
                        "compared": list(full.keys())})
        del full, resumed
        gc.collect()
    write_json(output / "recovery.json", {"passed": True, "device": device, "config": asdict(config), "results": results})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("recovery", "run"), required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/training_main.yaml")
    parser.add_argument("--data", type=Path, default=ROOT / "artifacts/corpus_100m_v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--variant", choices=("baseline", "helixdepth"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--recovery-child", action="store_true")
    args = parser.parse_args()
    if args.mode == "recovery":
        recovery(args.config, args.data, args.output, args.device)
    else:
        if not args.variant:
            parser.error("run requires --variant")
        config = TrainingConfig.from_yaml(args.config)
        train_run(args.variant, config, args.data, args.output, args.stop_after or config.schedule_steps,
                  args.resume, args.device, not args.recovery_child)


if __name__ == "__main__":
    main()
