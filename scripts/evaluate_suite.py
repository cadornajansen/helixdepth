"""Run isolated evaluation workers with bounded retries and verified resumption."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any, BinaryIO, Iterator


ROOT = Path(__file__).resolve().parents[1]
TASKS = ("hellaswag", "arc_easy", "piqa", "winogrande", "wikitext103")
SAMPLES = {"hellaswag": 10042, "arc_easy": 2376, "piqa": 1838, "winogrande": 1267}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


@contextmanager
def exclusive_lock(path: Path) -> Iterator[BinaryIO]:
    """OS lock releases on exit/crash; never unlink its inode while in use."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            if path.stat().st_size == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield handle
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            # POSIX close releases the lock after any worker inheriting it exits.


def stop_worker(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            process.terminate()
        else:
            os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            process.kill()
        else:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    except ProcessLookupError:
        process.wait()


def run_worker(command: list[str], log: Path, timeout: float,
               lock_fd: int | None = None, heartbeat: float = 30) -> dict[str, Any]:
    environment = dict(os.environ, TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="1")
    started = time.monotonic()
    options: dict[str, Any] = {}
    if os.name != "nt":
        options = {"start_new_session": True, "pass_fds": () if lock_fd is None else (lock_fd,)}
    with log.open("wb") as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT, **options)
        try:
            while True:
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    stop_worker(process)
                    return {"status": "timeout", "returncode": process.returncode}
                try:
                    code = process.wait(timeout=min(heartbeat, remaining))
                    return {"status": "exited", "returncode": code}
                except subprocess.TimeoutExpired:
                    print(f"Still running ({int(time.monotonic() - started)}s); log: {log}", flush=True)
        except BaseException:
            stop_worker(process)
            raise


def verify_result(directory: Path, task: str, identity: dict[str, Any]) -> None:
    """An exit code alone is insufficient: require complete, matching evidence."""
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    for key in ("model_sha256", "tokenizer_sha256", "batch_size"):
        if summary[key] != identity[key]:
            raise ValueError(f"Result identity mismatch: {key}")
    if summary["limit"] is not None or summary["zero_shot"] is not True:
        raise ValueError("A debug/few-shot result cannot complete the suite")
    if summary["tasks"] != ([] if task == "wikitext103" else [task]):
        raise ValueError("Unexpected worker task list")
    if summary["harness_version"] != "0.4.13" or summary["context"] != 512:
        raise ValueError("Unexpected harness version or context")
    filename = f"{task}.json"
    info = summary["files"][filename]
    result_path = directory / filename
    if result_path.stat().st_size != info["bytes"] or sha256(result_path) != info["sha256"]:
        raise ValueError("Result checksum mismatch")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if task == "wikitext103":
        if result["revision"] != identity["wikitext_revision"] or result["scored_tokens"] != 262144:
            raise ValueError("WikiText revision or slice mismatch")
        if result["split"] != "test" or not math.isfinite(result["token_perplexity"]) or result["token_perplexity"] < 1:
            raise ValueError("Invalid WikiText evaluation")
    else:
        counts = result["n-samples"][task]
        if counts["original"] != SAMPLES[task] or counts["effective"] != SAMPLES[task]:
            raise ValueError(f"Unexpected sample count for {task}: {counts}")
        if len(result["samples"][task]) != SAMPLES[task] or result["configs"][task]["num_fewshot"] != 0:
            raise ValueError("Missing samples or unexpected few-shot setting")
        accuracy = result["results"][task]["acc,none"]
        if not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
            raise ValueError("Invalid accuracy")


def cached_result(job: Path, task: str, identity: dict[str, Any]) -> Path | None:
    try:
        receipt = json.loads((job / "completed.json").read_text(encoding="utf-8"))
        name = receipt["attempt"]
        if not re.fullmatch(r"attempt-\d+", name):
            raise ValueError("Invalid attempt name")
        directory = job / name
        if sha256(directory / "summary.json") != receipt["summary_sha256"]:
            raise ValueError("Completion summary changed")
        verify_result(directory, task, identity)
        return directory
    except (OSError, ValueError, KeyError, TypeError):
        return None


def worker_command(model: Path, tokenizer: Path, output: Path, task: str,
                   identity: dict[str, Any]) -> list[str]:
    command = [sys.executable, "-u", str(ROOT / "scripts/evaluate_model.py"),
               "--model", str(model), "--tokenizer", str(tokenizer), "--output", str(output),
               "--device", identity["device"], "--batch-size", str(identity["batch_size"]),
               "--wikitext-revision", identity["wikitext_revision"]]
    return command + (["--wiki-only"] if task == "wikitext103" else ["--tasks", task, "--skip-wikitext"])


def run_job(job: Path, task: str, model: Path, tokenizer: Path,
            identity: dict[str, Any], attempts: int, timeout: float, lock_fd: int | None) -> dict[str, Any]:
    job.mkdir(parents=True, exist_ok=True)
    cached = cached_result(job, task, identity)
    if cached is not None:
        print(f"VERIFIED/SKIPPED {job.name}: {cached}", flush=True)
        return {"status": "passed", "output": str(cached)}
    failures: list[dict[str, Any]] = []
    previous = [int(p.stem.split("-")[1]) for p in job.iterdir() if re.fullmatch(r"attempt-\d+(\.log)?", p.name)]
    number = max(previous, default=0)
    for retry in range(attempts):
        number += 1
        directory = job / f"attempt-{number}"
        log = job / f"attempt-{number}.log"
        print(f"START {job.name}, try {retry + 1}/{attempts}; log: {log}", flush=True)
        command = worker_command(model, tokenizer, directory, task, identity)
        outcome = run_worker(command, log, timeout, lock_fd)
        outcome.update({"command": command, "log": str(log)})
        if outcome["status"] == "exited" and outcome["returncode"] == 0:
            try:
                verify_result(directory, task, identity)
                write_json(job / "completed.json", {"attempt": directory.name,
                           "summary_sha256": sha256(directory / "summary.json")})
                print(f"PASSED {job.name}", flush=True)
                return {"status": "passed", "output": str(directory)}
            except (OSError, ValueError, KeyError, TypeError) as error:
                outcome["validation_error"] = str(error)
        failures.append(outcome)
        write_json(job / f"attempt-{number}.status.json", outcome)
        write_json(job / "failures.json", {"attempts_this_invocation": failures})
        print(f"FAILED {job.name}: {outcome}; remaining tasks will still run.", flush=True)
    return {"status": "failed", "failures": failures}


def prepare_manifest(output: Path, identity: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "run.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != identity:
            raise ValueError("Run inputs/settings/source changed; preserve this run and choose a new --output")
    elif any(output.iterdir()):
        raise ValueError("Nonempty output lacks a run manifest; choose a new --output")
    else:
        write_json(path, identity)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/evaluation_recovery")
    parser.add_argument("--model-root", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "artifacts/cp2/tokenizer.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--attempts", type=int, choices=(1, 2), default=2)
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument("--wikitext-revision", default="b08601e04326c79dfdd32d625aee71d232d685c3")
    args = parser.parse_args()
    if args.batch_size < 1 or args.timeout_seconds < 1:
        parser.error("Batch size and timeout must be positive")
    models = {name: (args.model_root / f"main_{name}/model.pt").resolve() for name in ("baseline", "helixdepth")}
    sources = ["scripts/evaluate_model.py", "scripts/evaluate_suite.py"]
    sources += [str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT / "src/helixdepth").glob("*.py")]
    output = args.output.resolve()
    with exclusive_lock(ROOT / "artifacts/evaluation_suite.lock") as handle:
        identity = {"format": 1, "models": {name: sha256(path) for name, path in models.items()},
                    "tokenizer_sha256": sha256(args.tokenizer), "batch_size": args.batch_size,
                    "device": args.device, "wikitext_revision": args.wikitext_revision,
                    "sources": {name: sha256(ROOT / name) for name in sorted(sources)},
                    "packages": {name: importlib.metadata.version(name) for name in
                                 ("lm_eval", "torch", "datasets", "tokenizers", "PyYAML", "numpy")}}
        if identity["packages"]["lm_eval"] != "0.4.13":
            raise ValueError("Use the pinned evaluation dependencies: lm_eval==0.4.13")
        prepare_manifest(output, identity)
        report: dict[str, Any] = {"complete": False, "jobs": {}}
        write_json(output / "suite.json", report)
        for variant, model in models.items():
            model_identity = dict(identity, model_sha256=identity["models"][variant])
            for task in TASKS:
                name = f"{variant}_{task}"
                report["jobs"][name] = run_job(output / name, task, model, args.tokenizer.resolve(),
                    model_identity, args.attempts, args.timeout_seconds, handle.fileno())
                write_json(output / "suite.json", report)
        report["complete"] = all(row["status"] == "passed" for row in report["jobs"].values())
        write_json(output / "suite.json", report)
        print("ALL 10 EVALUATIONS VERIFIED" if report["complete"] else
              f"INCOMPLETE: inspect {output / 'suite.json'}; rerun this command to retry unfinished jobs.", flush=True)
        if not report["complete"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
