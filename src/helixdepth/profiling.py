"""Bounded full-model CUDA sweeps; only complete training updates are timed."""
import gc
import json
import math
import statistics
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch

from .config import ModelConfig
from .training import Trainer, TrainingConfig


def save_report(output: Path, report: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)


def measure_case(model_config: ModelConfig, config: TrainingConfig, data: Path,
                 warmup: int, steps: int, windows: int) -> dict[str, Any]:
    trainer = None
    phase = "setup"
    result: dict[str, Any] = {"variant": model_config.variant, "batch_size": config.batch_size,
                              "model_config": model_config.to_dict(), "training_config": asdict(config),
                              "warmup_steps": warmup, "steps_per_window": steps, "windows": []}
    try:
        torch.cuda.reset_peak_memory_stats()
        trainer = Trainer(model_config, config, data, "cuda")
        required = (warmup + steps * windows) * config.batch_size * 512 + 1
        if required > len(trainer.train_data.tokens):
            raise ValueError("Profile exceeds single-pass corpus; reduce batch/steps, do not cycle data")
        result["data_identity"] = trainer.train_data.identity
        phase = "warmup"
        for _ in range(warmup):
            trainer.train_step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        phase = "measurement"
        result["start_offset"] = trainer.train_data.offset
        for _ in range(windows):
            torch.cuda.synchronize()
            offset = trainer.train_data.offset
            started = time.perf_counter()
            for _ in range(steps):
                trainer.train_step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            tokens = trainer.train_data.offset - offset
            result["windows"].append({"seconds": elapsed, "target_tokens": tokens,
                                       "tokens_per_second": tokens / elapsed})
        durations = [window["seconds"] for window in result["windows"]]
        rates = [window["tokens_per_second"] for window in result["windows"]]
        tokens = sum(window["target_tokens"] for window in result["windows"])
        coefficient = statistics.pstdev(rates) / statistics.mean(rates)
        finite = all(math.isfinite(float(row[key])) for row in trainer.history
                     for key in ("loss", "gradient_norm_before_clip"))
        phase = "post_measurement_verification"
        finite_parameters = all(bool(torch.isfinite(parameter).all()) for parameter in trainer.model.parameters())
        if not finite or not finite_parameters:
            raise ValueError("Non-finite loss/gradient metric")
        result.update(status="passed", target_tokens=tokens, wall_seconds=sum(durations),
                      training_tokens_per_second=tokens / sum(durations), timing_cv=coefficient,
                      timing_stable=windows >= 3 and coefficient <= 0.10 and min(durations) >= 5.0,
                      stability_rule="At least 3 windows, each >=5 seconds, throughput CV <=10%",
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(), finite_losses_and_gradients=True,
                      end_offset=trainer.train_data.offset,
                      final_loss=trainer.history[-1]["loss"],
                      parameters=sum(parameter.numel() for parameter in trainer.model.parameters()))
    except torch.cuda.OutOfMemoryError as error:
        result.update(status="out_of_memory", failure_phase=phase, error=str(error))
    except (ValueError, RuntimeError, StopIteration) as error:
        result.update(status="failed", failure_phase=phase, error=str(error))
    finally:
        if trainer is not None:
            result["completed_steps"] = trainer.step
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
    return result


def profile_sweep(config: TrainingConfig, data: Path, variants: list[str], output: Path,
                  model_directory: Path, batches: list[int], warmup: int = 20,
                  steps: int = 100, windows: int = 3) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. GPU profiling was not started; use a CUDA-enabled PyTorch build on an authorized GPU.")
    if output.exists():
        raise FileExistsError(f"Preserving existing profile: {output}")
    if min(warmup, steps, windows) < 1 or not batches or any(batch < 1 for batch in batches):
        raise ValueError("Warmup, windows, steps and batch sizes must be positive")
    if batches != sorted(set(batches)):
        raise ValueError("Batch sizes must be unique and ascending")
    report: dict[str, Any] = {"profile_only": True, "context": 512, "precision": "float32",
        "device": torch.cuda.get_device_name(), "torch": str(torch.__version__),
        "cuda_version": torch.version.cuda, "runs": [], "final_training_token_budget": None,
        "includes": "Batch read/transfer, forward/loss/backward, finite checks, clipping, optimizer/scheduler, metric collection",
        "excludes": ["setup/model construction", "warmup", "validation", "checkpoint writes", "report writes"],
        "attention": "default SDPA; deterministic=False; TF32=False",
        "seed": config.seed,
        "order": "Every case starts from the configured seed and offset zero; identical offsets for both variants at each batch size"}
    save_report(output, report)
    for variant in variants:
        model_config = ModelConfig.from_yaml(model_directory / f"{variant}.yaml")
        if model_config.max_context != 512:
            raise ValueError("Full-model configs must have context 512")
        for batch in batches:
            case_config = replace(config, context=512, batch_size=batch, deterministic=False,
                                  warmup_steps=warmup, schedule_steps=warmup + steps * windows)
            result = measure_case(model_config, case_config, data, warmup, steps, windows)
            report["runs"].append(result)
            save_report(output, report)
            print(f"{variant} batch {batch}: {result['status']}; tokens/sec={result.get('training_tokens_per_second')}; stable={result.get('timing_stable')}", flush=True)
            if result["status"] != "passed":
                # Larger batches cannot establish reliability after this failure.
                report["runs"].extend({"variant": variant, "batch_size": larger, "status": "skipped",
                                        "reason": f"Previous batch {batch}: {result['status']}"}
                                       for larger in batches if larger > batch)
                save_report(output, report)
                break
