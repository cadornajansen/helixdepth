import json
import shutil
import time
from array import array
from dataclasses import replace
from pathlib import Path
from typing import Literal

import pytest
import torch

from helixdepth.config import tiny_config
from helixdepth.expansion import expand_corpus, read_rows
from helixdepth.packing import sha256_file
from helixdepth.training import Trainer, TrainingConfig
from scripts.prepare_continuation_data import build
from scripts.run_continuation import initialize, save_candidate, run
from scripts.train import assert_identical
from scripts.verify_continuation_backup import verify
from test_expansion import corpus
from test_training import packed_directory


def test_new_data_excludes_entire_previous_corpus_and_audit(corpus, tmp_path: Path) -> None:
    base, evidence, rows, limits = corpus
    old = tmp_path / "old"
    expand_corpus(rows, base, evidence, old, limits)
    fresh = [{"id": f"fresh-{i}", "text": f"New training material sample {i}.",
              "url": "https://example.com"} for i in range(100)]
    output = tmp_path / "fresh"
    report = build(rows + fresh, old, output, max_tokens=2000, candidates=180, minimum_tokens=1)
    assert report["fresh_verified"]
    old_hashes = {r["text_sha256"] for split in ("train", "validation") for r in read_rows(old / f"{split}.jsonl")}
    assert all(r["text_sha256"] not in old_hashes for r in read_rows(output / "train.jsonl"))
    assert (old / "validation.bin").read_bytes() == (output / "validation.bin").read_bytes()
    newer = [{"id": f"newer-{i}", "text": f"Distinct second continuation document {i}.",
              "url": "https://example.com"} for i in range(100)]
    second = tmp_path / "second"
    second_report = build(rows + fresh + newer, old, second, max_tokens=2000,
                          candidates=280, minimum_tokens=1, previous=output)
    prior_hashes = {r["text_sha256"] for r in read_rows(output / "selection_audit.jsonl")}
    assert all(r["text_sha256"] not in prior_hashes for r in read_rows(second / "train.jsonl"))
    assert second_report["previous_manifest_sha256"] == sha256_file(output / "manifest.json")
    with pytest.raises(FileExistsError):
        build(fresh, old, output)


def test_stage_loads_weights_resets_optimizer_and_recovers(packed_directory: Path, tmp_path: Path) -> None:
    config = TrainingConfig(context=8, batch_size=2, schedule_steps=6, warmup_steps=2)
    parent = Trainer(tiny_config("helixdepth"), config, packed_directory)
    parent.train_step()
    parent.save_checkpoint(tmp_path / "parent.pt")
    saved = torch.load(tmp_path / "parent.pt", weights_only=True)
    fresh_directory = tmp_path / "fresh_stage"
    fresh_directory.mkdir()
    for name in ("train.bin", "validation.bin", "manifest.json"):
        shutil.copyfile(packed_directory / name, fresh_directory / name)
    packed_directory = fresh_directory
    path = packed_directory / "train.bin"
    data = bytearray(path.read_bytes())
    data[0:4] = (2).to_bytes(4, "little")
    path.write_bytes(data)
    manifest_path = packed_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["splits"]["train"]["bin_sha256"] = sha256_file(path)
    manifest_path.write_text(json.dumps(manifest))
    stage = Trainer(tiny_config("helixdepth"), config, packed_directory)
    initialize(stage, saved)
    assert stage.step == 0 and not stage.optimizer.state
    assert_identical(stage.model.state_dict(), saved["model"])
    stage.train_step()
    stage.save_checkpoint(tmp_path / "stage.pt")
    stage.train_step()
    stage.save_checkpoint(tmp_path / "expected.pt")
    recovered = Trainer(tiny_config("helixdepth"), config, packed_directory)
    recovered.load_checkpoint(tmp_path / "stage.pt")
    recovered.train_step()
    recovered.save_checkpoint(tmp_path / "actual.pt")
    assert_identical(torch.load(tmp_path / "expected.pt", weights_only=True),
                     torch.load(tmp_path / "actual.pt", weights_only=True))
    save_candidate(recovered, tmp_path, 3.5)
    assert torch.load(tmp_path / "best.pt", weights_only=True)["selection"]["stage_step"] == 2
    saved["validation_identity"]["bin_sha256"] = "wrong"
    with pytest.raises(ValueError, match="validation"):
        initialize(recovered, saved)


def test_expired_deadline_refuses_before_loading(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="deadline"):
        run(tmp_path / "absent", tmp_path, tmp_path / "out", "cpu", 0)


def test_backup_verification_rejects_corrupt_download(tmp_path: Path) -> None:
    for variant in ("baseline", "helixdepth"):
        directory = tmp_path / f"{variant}_additional_100m_v1"
        directory.mkdir()
        files = {}
        for name in ("model.pt", "latest.pt", "best.pt"):
            path = directory / name
            path.write_bytes(b"checkpoint fixture")
            files[name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        (directory / "metrics.json").write_text(json.dumps({"stop_reason": "schedule_complete", "files": files}))
    assert verify(tmp_path)["verified"]
    (tmp_path / "helixdepth_additional_100m_v1/latest.pt").write_bytes(b"partial download")
    with pytest.raises(ValueError, match="Backup mismatch"):
        verify(tmp_path)


@pytest.mark.parametrize("variant", ["baseline", "helixdepth"])
def test_production_runner_selects_exports_and_resumes(
    packed_directory: Path, tmp_path: Path, variant: Literal["baseline", "helixdepth"]
) -> None:
    model_config = replace(tiny_config(variant), max_context=512)
    parent = Trainer(model_config, TrainingConfig(context=8, schedule_steps=6), packed_directory)
    parent.save_checkpoint(tmp_path / "parent.pt")
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    shutil.copyfile(packed_directory / "validation.bin", fresh / "validation.bin")
    with (fresh / "train.bin").open("wb") as stream:
        array("i", [i % 32 for i in range(8192 * 3 + 1)]).tofile(stream)
    manifest = json.loads((packed_directory / "manifest.json").read_text())
    manifest["fresh_verified"] = True
    manifest["splits"]["train"] = {"tokens": 8192 * 3 + 1, "bin_sha256": sha256_file(fresh / "train.bin")}
    manifest["files"] = {name: {"sha256": sha256_file(fresh / name)} for name in ("train.bin", "validation.bin")}
    (fresh / "manifest.json").write_text(json.dumps(manifest))
    deadline = time.time() + 600
    output = tmp_path / "continued"
    run(tmp_path / "parent.pt", fresh, output, "cpu", deadline, stop_after=1)
    assert json.loads((output / "metrics.json").read_text())["stop_reason"] == "requested_stop"
    run(tmp_path / "parent.pt", fresh, output, "cpu", deadline, resume=True)
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["stage"] == f"{variant} additional training"
    assert metrics["step"] == 3 and metrics["stage_targets"] == 24576
    assert sha256_file(output / "model.pt") == metrics["files"]["best.pt"]["sha256"]
    full = tmp_path / "full"
    run(tmp_path / "parent.pt", fresh, full, "cpu", deadline)
    assert_identical(torch.load(output / "latest.pt", weights_only=True),
                     torch.load(full / "latest.pt", weights_only=True))
    # A different packed corpus is required for the next stage.
    next_data = tmp_path / "next_data"
    shutil.copytree(fresh, next_data)
    with (next_data / "train.bin").open("wb") as stream:
        array("i", [(i + 1) % 32 for i in range(8192 * 3 + 1)]).tofile(stream)
    manifest["splits"]["train"]["bin_sha256"] = sha256_file(next_data / "train.bin")
    manifest["files"]["train.bin"]["sha256"] = sha256_file(next_data / "train.bin")
    (next_data / "manifest.json").write_text(json.dumps(manifest))
    next_output = tmp_path / "next_stage"
    run(output / "latest.pt", next_data, next_output, "cpu", deadline,
        stop_after=1, parent_report=output / "metrics.json")
    next_metrics = json.loads((next_output / "metrics.json").read_text())
    assert next_metrics["cumulative_targets"] == metrics["cumulative_targets"] + 8192
