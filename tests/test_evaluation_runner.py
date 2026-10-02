"""Failure injection for evaluation recovery; never downloads datasets or uses CUDA."""
import json
from pathlib import Path
from typing import Any, TextIO
import subprocess
import sys

import pytest

from scripts import evaluate_suite as runner


@pytest.fixture
def identity() -> dict[str, Any]:
    return {"model_sha256": "model", "tokenizer_sha256": "tokenizer", "batch_size": 16,
            "device": "cpu", "wikitext_revision": "pinned"}


@pytest.fixture
def fake_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    script = tmp_path / "worker.py"
    script.write_text('''import hashlib, json, os, sys
from pathlib import Path
output, task, identity, mode = Path(sys.argv[1]), sys.argv[2], json.loads(sys.argv[3]), sys.argv[4]
output.mkdir()
counter = output.parent / "launches.txt"
launches = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(str(launches))
if mode == "crash" or (mode == "crash_once" and launches == 1):
    print("simulated native-crash exit", flush=True)
    os._exit(139)
count = {"hellaswag": 10042, "arc_easy": 2376, "piqa": 1838, "winogrande": 1267}[task]
result = {"n-samples": {task: {"original": count, "effective": count}},
          "samples": {task: [{}] * count}, "configs": {task: {"num_fewshot": 0}},
          "results": {task: {"acc,none": 0.5}}}
path = output / (task + ".json")
path.write_text(json.dumps(result))
summary = {**identity, "limit": None, "zero_shot": True, "tasks": [task], "harness_version": "0.4.13",
           "context": 512, "files": {path.name: {"bytes": path.stat().st_size,
           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}}
(output / "summary.json").write_text(json.dumps(summary))
''', encoding="utf-8")
    monkeypatch.setattr(runner, "worker_command", lambda model, tokenizer, output, task, identity:
                        [sys.executable, str(script), str(output), task, json.dumps(identity), "crash_once"])
    return script


def test_crash_retry_resume_and_corrupt_result(tmp_path: Path, identity: dict[str, Any], fake_worker: Path) -> None:
    job = tmp_path / "baseline_arc_easy"
    args = (job, "arc_easy", tmp_path / "model", tmp_path / "tokenizer", identity, 2, 10, None)
    result = runner.run_job(*args)
    assert result["status"] == "passed"
    assert (job / "launches.txt").read_text() == "2"
    assert "simulated native-crash" in (job / "attempt-1.log").read_text()
    assert runner.run_job(*args) == result
    assert (job / "launches.txt").read_text() == "2"  # No extra worker on resume.
    (Path(result["output"]) / "arc_easy.json").write_text("truncated")
    repaired = runner.run_job(*args)
    assert repaired["status"] == "passed"
    assert repaired["output"] != result["output"]
    assert (job / "launches.txt").read_text() == "3"
    assert (Path(result["output"]) / "arc_easy.json").read_text() == "truncated"


def test_persistent_failure_stops_retrying_and_next_job_runs(
    tmp_path: Path, identity: dict[str, Any], fake_worker: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "worker_command", lambda model, tokenizer, output, task, identity:
                        [sys.executable, str(fake_worker), str(output), task, json.dumps(identity),
                         "crash" if task == "arc_easy" else "success"])
    job = tmp_path / "failed"
    assert runner.run_job(job, "arc_easy", tmp_path, tmp_path, identity, 2, 10, None)["status"] == "failed"
    assert (job / "launches.txt").read_text() == "2"
    assert not (job / "completed.json").exists()
    assert runner.run_job(tmp_path / "next", "piqa", tmp_path, tmp_path, identity, 2, 10, None)["status"] == "passed"


def test_timeout_terminates_worker(tmp_path: Path) -> None:
    result = runner.run_worker([sys.executable, "-c", "import time; time.sleep(60)"],
                               tmp_path / "timeout.log", timeout=0.2, heartbeat=0.1)
    assert result["status"] == "timeout"
    assert result["returncode"] is not None


def test_cross_process_lock_released_after_exit(tmp_path: Path) -> None:
    lock = tmp_path / "lock"
    command = [sys.executable, "-c", "from pathlib import Path; import sys; "
               "from scripts.evaluate_suite import exclusive_lock; "
               "lock=exclusive_lock(Path(sys.argv[1])); lock.__enter__()", str(lock)]
    with runner.exclusive_lock(lock):
        assert subprocess.run(command, cwd=runner.ROOT, capture_output=True).returncode != 0
    assert subprocess.run(command, cwd=runner.ROOT, capture_output=True).returncode == 0


def test_manifest_refuses_changed_inputs_and_untracked_outputs(tmp_path: Path) -> None:
    output = tmp_path / "run"
    runner.prepare_manifest(output, {"model": "first"})
    runner.prepare_manifest(output, {"model": "first"})
    with pytest.raises(ValueError, match="changed"):
        runner.prepare_manifest(output, {"model": "second"})
    assert json.loads((output / "run.json").read_text()) == {"model": "first"}
    orphan = tmp_path / "orphan"
    orphan.mkdir()
    (orphan / "result.json").write_text("{}")
    with pytest.raises(ValueError, match="Nonempty"):
        runner.prepare_manifest(orphan, {})


def test_partial_json_never_replaces_complete_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "result.json"
    runner.write_json(path, {"valid": True})
    def interrupted_dump(value: dict[str, Any], stream: TextIO, **kwargs: object) -> None:
        stream.write('{"partial":')
        raise OSError("Interrupted write")
    monkeypatch.setattr(runner.json, "dump", interrupted_dump)
    with pytest.raises(OSError):
        runner.write_json(path, {"valid": False})
    assert json.loads(path.read_text()) == {"valid": True}


def test_reject_wrong_model_incomplete_and_nonfinite_results(
    tmp_path: Path, identity: dict[str, Any], fake_worker: Path,
) -> None:
    result = runner.run_job(tmp_path / "job", "piqa", tmp_path, tmp_path, identity, 2, 10, None)
    directory = Path(result["output"])
    with pytest.raises(ValueError, match="identity"):
        runner.verify_result(directory, "piqa", dict(identity, model_sha256="different"))
    path = directory / "piqa.json"
    summary_path = directory / "summary.json"
    original = json.loads(path.read_text())
    for change in ("partial", "nonfinite"):
        data = json.loads(json.dumps(original))
        if change == "partial":
            data["n-samples"]["piqa"]["effective"] = 1
        else:
            data["results"]["piqa"]["acc,none"] = float("nan")
        path.write_text(json.dumps(data))
        summary = json.loads(summary_path.read_text())
        summary["files"][path.name] = {"sha256": runner.sha256(path), "bytes": path.stat().st_size}
        runner.write_json(summary_path, summary)
        with pytest.raises(ValueError):
            runner.verify_result(directory, "piqa", identity)


def test_official_task_discovery_is_restricted() -> None:
    pytest.importorskip("lm_eval")
    from scripts.evaluate_model import TASK_DIRS, task_manager
    for task in TASK_DIRS:
        manager = task_manager(task)
        assert task in manager.all_tasks
        assert len(manager.all_tasks) <= 5
        assert "mmlu" not in manager.all_tasks


def test_real_harness_adapter_with_offline_arc_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    harness = pytest.importorskip("lm_eval")
    import datasets
    from tokenizers import Tokenizer, models, pre_tokenizers
    from helixdepth.config import tiny_config
    from helixdepth.model import LanguageModel
    from scripts.evaluate_model import HelixHarness, task_manager, write_json

    data = datasets.Dataset.from_list([
        {"question": "Which is blue?", "choices": {"label": ["A", "B"], "text": ["sky", "grass"]}, "answerKey": "A"},
        {"question": "Which is green?", "choices": {"label": ["A", "B"], "text": ["sky", "grass"]}, "answerKey": "B"},
    ])
    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs:
                        datasets.DatasetDict(train=data, validation=data, test=data))
    tokenizer = Tokenizer(models.WordLevel({"<unk>": 0, "sky": 4, "grass": 5}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    adapter = HelixHarness(LanguageModel(tiny_config("baseline")), tokenizer, batch_size=2)
    result = harness.simple_evaluate(model=adapter, tasks=["arc_easy"], task_manager=task_manager("arc_easy"),
                                    num_fewshot=0, log_samples=True, bootstrap_iters=0)
    assert result["n-samples"]["arc_easy"] == {"original": 2, "effective": 2}
    assert result["configs"]["arc_easy"]["num_fewshot"] == 0
    assert len(result["samples"]["arc_easy"]) == 2
    write_json(tmp_path / "fixture.json", result)
    assert json.loads((tmp_path / "fixture.json").read_text())["results"]["arc_easy"]["acc,none"] >= 0
