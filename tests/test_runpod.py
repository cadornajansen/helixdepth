import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from helixdepth import profiling
from helixdepth.config import ModelConfig
from helixdepth.training import TrainingConfig
from scripts.verify_transfer import sha256, verify_files


def test_transfer_rejects_corruption_and_missing_files(tmp_path: Path) -> None:
    path = tmp_path / "data.bin"
    path.write_bytes(b"verified")
    manifest = {"files": {"data.bin": {"bytes": 8, "sha256": sha256(path)}}}
    assert verify_files(tmp_path, manifest) == 1
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        verify_files(tmp_path, manifest)
    path.unlink()
    with pytest.raises(ValueError, match="Missing"):
        verify_files(tmp_path, manifest)


def test_transfer_rejects_path_escape(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsafe"):
        verify_files(tmp_path, {"files": {"../outside": {"bytes": 1, "sha256": "invalid"}}})


@pytest.fixture
def fake_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "test double, not GPU evidence")
    for name in ("synchronize", "reset_peak_memory_stats", "empty_cache"):
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 123)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 456)


def test_profile_times_only_training_and_counts_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_cuda: None
) -> None:
    class FakeTrainer:
        def __init__(self, *args: Any) -> None:
            self.step = 0
            self.history: list[dict[str, float]] = []
            self.train_data = SimpleNamespace(tokens=range(10000), offset=0, identity={"fixture": True})
            self.model = SimpleNamespace(parameters=lambda: [torch.tensor(1.0)])

        def train_step(self) -> None:
            self.step += 1
            self.train_data.offset += 512
            self.history.append({"loss": 1.0, "gradient_norm_before_clip": 1.0})

        def validate(self, *args: Any) -> None:
            raise AssertionError("Profiling must not validate")

        def save_checkpoint(self, *args: Any) -> None:
            raise AssertionError("Profiling must not write checkpoints")

    monkeypatch.setattr(profiling, "Trainer", FakeTrainer)
    ticks = iter([0.0, 10.0, 10.0, 20.0, 20.0, 30.0])
    monkeypatch.setattr(profiling.time, "perf_counter", lambda: next(ticks))
    result = profiling.measure_case(ModelConfig(), TrainingConfig(), tmp_path, 2, 2, 3)
    assert result["target_tokens"] == 6 * 512
    assert result["wall_seconds"] == 30.0 and result["timing_stable"]
    assert result["start_offset"] == 1024 and result["end_offset"] == 4096
    assert result["completed_steps"] == 8


def test_sweep_preserves_oom_and_continues_other_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_cuda: None
) -> None:
    calls: list[tuple[str, int]] = []

    def measure(model: ModelConfig, config: TrainingConfig, *args: Any) -> dict[str, Any]:
        calls.append((model.variant, config.batch_size))
        status = "out_of_memory" if model.variant == "baseline" and config.batch_size == 2 else "passed"
        return {"variant": model.variant, "batch_size": config.batch_size, "status": status}

    monkeypatch.setattr(profiling, "measure_case", measure)
    output = tmp_path / "profile.json"
    profiling.profile_sweep(TrainingConfig(), tmp_path, ["baseline", "helixdepth"], output,
                            Path(__file__).resolve().parents[1] / "configs", [1, 2, 4])
    report = json.loads(output.read_text())
    assert calls == [("baseline", 1), ("baseline", 2), ("helixdepth", 1), ("helixdepth", 2), ("helixdepth", 4)]
    assert report["runs"][1]["status"] == "out_of_memory"
    assert report["runs"][2]["status"] == "skipped"


def test_numerical_failure_recorded_without_throughput(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_cuda: None
) -> None:
    class FailedTrainer:
        def __init__(self, *args: Any) -> None:
            self.step = 0
            self.train_data = SimpleNamespace(tokens=range(10000), identity={"fixture": True})

        def train_step(self) -> None:
            raise ValueError("Non-finite training loss")

    monkeypatch.setattr(profiling, "Trainer", FailedTrainer)
    result = profiling.measure_case(ModelConfig(), TrainingConfig(), tmp_path, 2, 2, 3)
    assert result["status"] == "failed" and "Non-finite" in result["error"]
    assert "training_tokens_per_second" not in result
