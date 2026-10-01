import json
import random
from array import array
from pathlib import Path
from typing import Literal

import pytest
import torch
from torch.nn import functional as F

from helixdepth.config import tiny_config
from helixdepth.model import LanguageModel
from helixdepth.packing import PackedStream, sha256_file
from helixdepth.training import Trainer, TrainingConfig, shifted_loss
from scripts.train import assert_identical, profile_gpu


@pytest.fixture
def packed_directory(tmp_path: Path) -> Path:
    manifest = {"format": "little-endian-int32", "tokenizer_sha256": "fixture", "splits": {}}
    for split, count in (("train", 121), ("validation", 18)):
        path = tmp_path / f"{split}.bin"
        with path.open("wb") as output:
            array("i", [index % 32 for index in range(count)]).tofile(output)
        manifest["splits"][split] = {"tokens": count, "bin_sha256": sha256_file(path)}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


def test_packing_shift_boundaries_order_and_exhaustion(packed_directory: Path) -> None:
    stream = PackedStream(packed_directory, "train", context=8, batch_size=2)
    inputs, targets = stream.peek()
    assert inputs.shape == targets.shape == (2, 8)
    assert torch.equal(inputs.flatten(), torch.arange(16))
    assert torch.equal(targets.flatten(), torch.arange(1, 17))
    assert stream.offset == 0
    stream.advance()
    later_inputs, later_targets = stream.peek()
    assert later_inputs[0, 0] == 16 and later_targets[0, 0] == 17
    # Token 16 was the previous final target and is the next input, not a repeated target.
    while stream.offset + 17 <= len(stream.tokens):
        stream.advance()
    with pytest.raises(StopIteration, match="Single-pass corpus exhausted"):
        stream.peek()


def test_packed_file_tampering_rejected(packed_directory: Path) -> None:
    with (packed_directory / "train.bin").open("r+b") as stream:
        stream.write(b"\xff")
    with pytest.raises(ValueError, match="differs from manifest"):
        PackedStream(packed_directory, "train", context=8, batch_size=1)


def test_explicit_targets_are_not_shifted_twice() -> None:
    model = LanguageModel(tiny_config("baseline")).eval()
    inputs = torch.arange(8).unsqueeze(0)
    targets = inputs + 1
    expected = F.cross_entropy(model(inputs).flatten(0, 1), targets.flatten())
    torch.testing.assert_close(shifted_loss(model, inputs, targets), expected, rtol=0, atol=0)


@pytest.mark.parametrize("variant", ["baseline", "helixdepth"])
def test_resume_matches_model_optimizer_schedule_rng_and_position(
    variant: Literal["baseline", "helixdepth"], packed_directory: Path, tmp_path: Path
) -> None:
    config = TrainingConfig(context=8, batch_size=2, schedule_steps=6, warmup_steps=2)
    model_config = tiny_config(variant)
    full = Trainer(model_config, config, packed_directory)
    for _ in range(6):
        full.train_step()
    full.save_checkpoint(tmp_path / "full.pt")
    expected_python, expected_torch = random.random(), torch.rand(3)
    interrupted = Trainer(model_config, config, packed_directory)
    for _ in range(3):
        interrupted.train_step()
    interrupted.save_checkpoint(tmp_path / "interrupted.pt")
    resumed = Trainer(model_config, config, packed_directory)
    random.random()
    torch.rand(7)
    resumed.load_checkpoint(tmp_path / "interrupted.pt")
    for _ in range(3):
        resumed.train_step()
    resumed.save_checkpoint(tmp_path / "resumed.pt")
    assert_identical(torch.load(tmp_path / "full.pt", weights_only=True),
                     torch.load(tmp_path / "resumed.pt", weights_only=True))
    assert random.random() == expected_python
    assert torch.equal(torch.rand(3), expected_torch)
    assert resumed.train_data.offset == 96
    assert not (tmp_path / "resumed.pt.tmp").exists()


def test_validation_counts_tail_and_preserves_training_state(packed_directory: Path) -> None:
    config = TrainingConfig(context=8, batch_size=1, schedule_steps=6, warmup_steps=2)
    trainer = Trainer(tiny_config("baseline"), config, packed_directory)
    rng = torch.get_rng_state().clone()
    position = trainer.train_data.state_dict()
    complete = trainer.validate()
    assert complete["scored_tokens"] == 17 and complete["batches"] == 3
    assert complete["complete_split"]
    partial = trainer.validate(max_batches=1)
    assert partial["scored_tokens"] == 8 and not partial["complete_split"]
    assert torch.isfinite(torch.tensor(complete["loss"]))
    assert trainer.train_data.state_dict() == position
    assert torch.equal(torch.get_rng_state(), rng)
    assert trainer.model.training


def test_resume_rejects_changed_data_or_settings(packed_directory: Path, tmp_path: Path) -> None:
    config = TrainingConfig(context=8, schedule_steps=6)
    trainer = Trainer(tiny_config("baseline"), config, packed_directory)
    trainer.train_step()
    trainer.save_checkpoint(tmp_path / "resume.pt")
    changed_config = Trainer(tiny_config("baseline"), TrainingConfig(context=8, schedule_steps=7), packed_directory)
    with pytest.raises(ValueError, match="training/runtime mismatch"):
        changed_config.load_checkpoint(tmp_path / "resume.pt")
    trainer.train_data.identity = {**trainer.train_data.identity, "bin_sha256": "changed"}
    with pytest.raises(ValueError, match="data/settings mismatch"):
        trainer.load_checkpoint(tmp_path / "resume.pt")


def test_cuda_profile_refuses_without_cuda(packed_directory: Path, tmp_path: Path,
                                          monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    output = tmp_path / "gpu_profile.json"
    with pytest.raises(RuntimeError, match="profiling was not started"):
        profile_gpu(TrainingConfig(), packed_directory, ["baseline"], output, 3, 10)
    assert not output.exists()


def test_failed_checkpoint_write_preserves_previous_checkpoint(
    packed_directory: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer = Trainer(tiny_config("baseline"), TrainingConfig(context=8, schedule_steps=6), packed_directory)
    trainer.train_step()
    path = tmp_path / "latest.pt"
    trainer.save_checkpoint(path)
    previous_hash = sha256_file(path)
    trainer.train_step()

    def interrupted_write(value: object, stream: object) -> None:
        raise OSError("Simulated incomplete checkpoint write")

    monkeypatch.setattr(torch, "save", interrupted_write)
    with pytest.raises(OSError, match="incomplete checkpoint write"):
        trainer.save_checkpoint(path)
    assert sha256_file(path) == previous_hash
    assert torch.load(path, weights_only=True)["step"] == 1


def test_gradient_clipping_enforces_norm_limit(packed_directory: Path) -> None:
    config = TrainingConfig(context=8, schedule_steps=6, gradient_clip=1e-4)
    trainer = Trainer(tiny_config("helixdepth"), config, packed_directory)
    metric = trainer.train_step()
    assert metric["gradient_norm_before_clip"] > config.gradient_clip
    squared_norm = sum(parameter.grad.square().sum() for parameter in trainer.model.parameters()
                       if parameter.grad is not None)
    assert squared_norm.sqrt().item() <= config.gradient_clip * 1.01
