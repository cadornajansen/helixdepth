"""Local reliability smoke checks and an explicit, future CUDA profiling command."""
import argparse
import gc
import json
import random
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Literal

import torch

from helixdepth.profiling import profile_sweep
from helixdepth.config import ModelConfig
from helixdepth.training import Trainer, TrainingConfig

ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def smoke_model_config(variant: Literal["baseline", "helixdepth"]) -> ModelConfig:
    return ModelConfig(variant=variant, vocab_size=16384, max_context=512, width=32,
                       heads=4, ffn_width=64, base_blocks=6,
                       stages=6 if variant == "baseline" else 12,
                       rank=4, depth_width=8, hyper_width=16)


def assert_identical(left: Any, right: Any, location: str = "checkpoint") -> None:
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or not torch.equal(left, right):
            raise AssertionError(f"Tensor mismatch: {location}")
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise AssertionError(f"Keys differ: {location}")
        for key in left:
            assert_identical(left[key], right[key], f"{location}.{key}")
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise AssertionError(f"Length mismatch: {location}")
        for index, (a, b) in enumerate(zip(left, right)):
            assert_identical(a, b, f"{location}[{index}]")
    elif left != right:
        raise AssertionError(f"Value mismatch: {location}")


def run_training(model_config: ModelConfig, config: TrainingConfig, data: Path,
                 destination: Path, stop_after: int, resume: Path | None = None,
                 device: str = "cpu") -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(f"Preserving previous run: {destination}; choose a new --output")
    trainer = Trainer(model_config, config, data, device)
    if resume is not None:
        trainer.load_checkpoint(resume)
    if not trainer.step < stop_after <= config.schedule_steps:
        raise ValueError("Stopping step must advance within the saved schedule")
    destination.mkdir(parents=True)
    initial_validation = trainer.validate(config.validation_batches)
    started = time.perf_counter()
    try:
        while trainer.step < stop_after:
            metric = trainer.train_step()
            print(f"{model_config.variant} step {trainer.step}: loss={metric['loss']:.6f}", flush=True)
            if trainer.step % config.checkpoint_every == 0:
                trainer.save_checkpoint(destination / "latest.pt")
    except KeyboardInterrupt:
        print(f"Interrupted; resume the last complete checkpoint in {destination}. Pending step is not saved.", flush=True)
        raise
    elapsed = time.perf_counter() - started
    final_validation = trainer.validate(config.validation_batches)
    trainer.save_checkpoint(destination / "latest.pt")
    report = {"variant": model_config.variant, "model_config": model_config.to_dict(),
              "training_config": asdict(config), "runtime": trainer.runtime_identity(),
              "steps": trainer.step, "target_tokens": trainer.train_data.offset,
              "initial_validation": initial_validation, "final_validation": final_validation,
              "history": trainer.history, "data_state": trainer.train_data.state_dict(),
              "validation_identity": trainer.validation_data.identity,
              "training_loop_seconds": elapsed, "resume_from": str(resume) if resume else None,
              "checkpoint": str(destination / "latest.pt"), "model_parameters": sum(p.numel() for p in trainer.model.parameters()),
              "scope": "Short reliability run; validation is partial and losses are not a benchmark result"}
    write_json(destination / "metrics.json", report)
    return report


def smoke_checks(config: TrainingConfig, data: Path, output: Path, report_path: Path) -> None:
    if not config.deterministic or config.context != 512 or config.schedule_steps < 4:
        raise ValueError("Smoke equivalence checks require deterministic context-512 settings and at least four steps")
    if output.exists():
        raise FileExistsError(f"Preserving previous smoke run: {output}")
    results: list[dict[str, Any]] = []
    for variant in ("baseline", "helixdepth"):
        model_config = smoke_model_config(variant)
        full_dir = output / variant / "uninterrupted"
        interrupted_dir = output / variant / "interrupted"
        resumed_dir = output / variant / "resumed"
        full = run_training(model_config, config, data, full_dir, config.schedule_steps)
        interruption_step = config.schedule_steps // 2
        run_training(model_config, config, data, interrupted_dir, interruption_step)
        config_path = output / "training_used.json"
        # YAML accepts JSON, preserving exactly the scheduler horizon during resume.
        write_json(config_path, asdict(config))
        command = [sys.executable, str(Path(__file__).resolve()), "--mode", "run", "--variant", variant,
                   "--model-size", "smoke", "--config", str(config_path), "--data", str(data),
                   "--output", str(resumed_dir), "--resume", str(interrupted_dir / "latest.pt")]
        subprocess.run(command, check=True)
        uninterrupted = torch.load(full_dir / "latest.pt", map_location="cpu", weights_only=True)
        resumed = torch.load(resumed_dir / "latest.pt", map_location="cpu", weights_only=True)
        assert_identical(uninterrupted, resumed)
        # Check that restored states also reproduce the *next* random draws and batch.
        trainer = Trainer(model_config, config, data)
        trainer.load_checkpoint(full_dir / "latest.pt")
        expected_python = random.random()
        expected_torch = torch.rand(4)
        expected_batch = trainer.train_data.peek()
        trainer.load_checkpoint(resumed_dir / "latest.pt")
        if random.random() != expected_python or not torch.equal(torch.rand(4), expected_torch):
            raise AssertionError("Restored random state diverges")
        actual_batch = trainer.train_data.peek()
        if any(not torch.equal(a, b) for a, b in zip(expected_batch, actual_batch)):
            raise AssertionError("Restored data position diverges")
        results.append({**full, "interruption_step": interruption_step,
                        "fresh_process_resume": True, "resume_exact_match": True,
                        "compared": ["model", "optimizer", "scheduler", "history", "random_states",
                                     "data_position", "next_random_draws", "next_batch"],
                        "resumed_checkpoint": str(resumed_dir / "latest.pt")})
        del trainer
        gc.collect()
    if results[0]["data_state"] != results[1]["data_state"]:
        raise AssertionError("Architectures did not consume identical data order")
    report = {"checkpoint": "CP3", "local_checks_passed": True, "runs": results,
              "identical_data_order": True, "tokenizer_fixed": True,
              "gpu_profile": "Prepared, not executed; no GPU rented",
              "final_training_token_budget": None,
              "limitations": ["Smoke models retain vocabulary/context/six stored blocks but reduce width/FFN/rank",
                              "Float32 only; full-size training and GPU resume are unvalidated",
                              "Validation loss covers only the configured validation prefix",
                              "Single-pass training; corpus expansion needed before main experiment"]}
    write_json(report_path, report)
    print("Both architectures: exact fresh-process resume and identical data order verified.", flush=True)


def profile_gpu(config: TrainingConfig, data: Path, variants: list[str], output: Path,
                warmup: int, measured_steps: int, batches: list[int] | None = None,
                windows: int = 3) -> None:
    profile_sweep(config, data, variants, output, ROOT / "configs",
                  batches or [config.batch_size], warmup, measured_steps, windows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "run", "profile"), default="smoke")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/training.yaml")
    parser.add_argument("--data", type=Path, default=ROOT / "artifacts/cp3_data")
    parser.add_argument("--variant", choices=("baseline", "helixdepth", "both"), default="both")
    parser.add_argument("--model-size", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path, default=ROOT / "docs/results/cp3_smoke.json")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--profile-steps", type=int, default=100)
    parser.add_argument("--profile-windows", type=int, default=3)
    parser.add_argument("--batch-sizes", type=int, nargs="+")
    args = parser.parse_args()
    config = TrainingConfig.from_yaml(args.config)
    if args.batch_size is not None:
        config = replace(config, batch_size=args.batch_size)
    if args.batch_size is not None and args.batch_sizes is not None:
        parser.error("Choose --batch-size or --batch-sizes, not both")
    if args.mode == "profile":
        if args.device != "cuda" or args.model_size != "full":
            parser.error("Profiling requires explicit --device cuda --model-size full")
        variants = ["baseline", "helixdepth"] if args.variant == "both" else [args.variant]
        profile_gpu(config, args.data, variants, args.output or ROOT / "docs/results/gpu_profile.json",
                    args.warmup, args.profile_steps, args.batch_sizes, args.profile_windows)
    elif args.mode == "smoke":
        if args.device != "cpu" or args.resume is not None or args.model_size != "smoke":
            parser.error("Reliability smoke checks run reduced models on CPU without --resume")
        smoke_checks(config, args.data, args.output or ROOT / "artifacts/cp3_smoke", args.report)
    else:
        if args.variant == "both" or args.output is None:
            parser.error("Run mode requires one --variant and an explicit --output")
        model_config = (smoke_model_config(args.variant) if args.model_size == "smoke" else
                        ModelConfig.from_yaml(ROOT / "configs" / f"{args.variant}.yaml"))
        run_training(model_config, config, args.data, args.output,
                     args.stop_after or config.schedule_steps, args.resume, args.device)


if __name__ == "__main__":
    main()
