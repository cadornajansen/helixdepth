"""Zero-shot lm-evaluation-harness and held-out WikiText-103 token perplexity."""
import argparse
import faulthandler
import hashlib
import importlib.metadata
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import lm_eval
import torch
from datasets import load_dataset
import lm_eval.tasks
from lm_eval.tasks import TaskManager
from lm_eval.api.model import LM
from lm_eval.api.instance import Instance
from tokenizers import Tokenizer

from helixdepth.evaluation import encode_pair, score_tokens
from helixdepth.model import LanguageModel
from helixdepth.packing import sha256_file as sha256


TASK_DIRS = {"hellaswag": "hellaswag", "arc_easy": "arc", "piqa": "piqa", "winogrande": "winogrande"}


def task_manager(task: str) -> TaskManager:
    """Load unchanged official definitions without scanning the whole catalogue."""
    directory = Path(lm_eval.tasks.__file__).parent / TASK_DIRS[task]
    manager = TaskManager(include_defaults=False, include_path=str(directory))
    if task not in manager.all_tasks:
        raise ValueError(f"Pinned harness does not provide {task} in {directory}")
    return manager


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Publish a complete JSON file, never an interrupted partial result."""
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, default=str, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class HelixHarness(LM):
    def __init__(self, model: LanguageModel, tokenizer: Tokenizer, batch_size: int = 16) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self._device = next(model.parameters()).device
        self.batch_size = batch_size

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        pairs = [encode_pair(self.tokenizer, *request.args) for request in requests]
        return score_tokens(self.model, pairs, self.batch_size)

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[float]:
        pairs = [([2], self.tokenizer.encode(request.args[0], add_special_tokens=False).ids) for request in requests]
        return [score for score, _ in score_tokens(self.model, pairs, self.batch_size)]

    def generate_until(self, requests: list[Instance]) -> list[str]:
        raise NotImplementedError("This evaluation adapter supports likelihood tasks only")


def wikitext(model: LanguageModel, tokenizer: Tokenizer, revision: str, batch_size: int) -> dict[str, Any]:
    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", revision=revision, split="test")
    # Explicit contiguous test slice, fixed before inspecting either model's scores.
    text = "\n".join(dataset["text"])
    all_tokens = tokenizer.encode(text, add_special_tokens=False).ids
    tokens = all_tokens[:262144]
    if len(tokens) != 262144:
        raise ValueError("WikiText test data is shorter than the frozen evaluation slice")
    logprob, _ = score_tokens(model, [([2], tokens)], batch_size)[0]
    loss = -logprob / len(tokens)
    return {"dataset": "Salesforce/wikitext", "configuration": "wikitext-103-raw-v1", "split": "test",
            "revision": revision, "rows": len(dataset), "joined_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "slice": "First 262144 own-tokenizer tokens from newline-joined test rows; no EOS appended",
            "available_tokens": len(all_tokens), "scored_tokens": len(tokens), "prefix_token_id": 2,
            "window": model.config.max_context, "mean_token_nll": loss, "token_perplexity": math.exp(loss),
            "note": "Own-tokenizer perplexity; not directly comparable across different tokenizers"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/cp2/tokenizer.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, help="Debug only: limits benchmark examples and marks report incomplete")
    parser.add_argument("--tasks", nargs="+", choices=("hellaswag", "arc_easy", "piqa", "winogrande"),
                        default=("hellaswag", "arc_easy", "piqa", "winogrande"),
                        help="Benchmarks to run; use one per process to isolate failures")
    parser.add_argument("--skip-wikitext", action="store_true",
                        help="Skip the separate held-out WikiText perplexity evaluation")
    parser.add_argument("--wiki-only", action="store_true", help="Run only the frozen WikiText test slice")
    parser.add_argument("--wikitext-revision", required=True)
    args = parser.parse_args()
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Batch size and example limit must be positive")
    if args.wiki_only and args.skip_wikitext:
        parser.error("--wiki-only and --skip-wikitext cannot be combined")
    if args.wiki_only:
        args.tasks = []
    if importlib.metadata.version("lm_eval") != "0.4.13":
        raise RuntimeError("Use the pinned evaluation dependencies: lm_eval==0.4.13")
    faulthandler.enable()
    if args.output.exists():
        raise FileExistsError(f"Preserving existing evaluation: {args.output}")
    args.output.mkdir(parents=True)
    print(f"Evaluation started: model={args.model}, output={args.output}", flush=True)
    torch.set_num_threads(1)
    torch.manual_seed(2026)
    print("Loading trained checkpoint...", flush=True)
    model = LanguageModel.load(args.model).to(args.device).eval()
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    adapter = HelixHarness(model, tokenizer, args.batch_size)
    print("Model and tokenizer ready.", flush=True)
    started = time.monotonic()
    for task in args.tasks:
        print(f"Starting {task}: loading task/data, then scoring (downloads may take time).", flush=True)
        manager = task_manager(task)
        print(f"Task discovery ready: {len(manager.all_tasks)} names from {TASK_DIRS[task]} only.", flush=True)
        result = lm_eval.simple_evaluate(model=adapter, tasks=[task], num_fewshot=0,
                    limit=args.limit, bootstrap_iters=1000, log_samples=True,
                    task_manager=manager,
                    random_seed=2026, numpy_random_seed=2026, torch_random_seed=2026, fewshot_random_seed=2026)
        if result is None or task not in result.get("results", {}):
            raise ValueError(f"Harness returned no result for {task}")
        write_json(args.output / f"{task}.json", result)
        print(task, result["results"], flush=True)
    wiki = None
    if not args.skip_wikitext:
        print("Starting held-out WikiText-103...", flush=True)
        wiki = wikitext(model, tokenizer, args.wikitext_revision, args.batch_size)
        write_json(args.output / "wikitext103.json", wiki)
    report = {"complete_benchmarks": args.limit is None and set(args.tasks) == {
                  "hellaswag", "arc_easy", "piqa", "winogrande"},
              "limit": args.limit, "seconds": time.monotonic() - started,
              "model_sha256": sha256(args.model), "tokenizer_sha256": sha256(args.tokenizer),
              "harness_version": importlib.metadata.version("lm_eval"), "torch": str(torch.__version__),
              "context": model.config.max_context,
              "batch_size": args.batch_size, "zero_shot": True, "wikitext": wiki,
              "tasks": list(args.tasks),
              "files": {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in args.output.glob("*.json")}}
    write_json(args.output / "summary.json", report)
    print(f"Evaluation complete: {args.output}", flush=True)


if __name__ == "__main__":
    main()
