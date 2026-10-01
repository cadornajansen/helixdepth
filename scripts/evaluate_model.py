"""Zero-shot lm-evaluation-harness and held-out WikiText-103 token perplexity."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import time
from pathlib import Path
from typing import Any

import lm_eval
import torch
from datasets import load_dataset
from huggingface_hub import HfApi
from lm_eval.api.model import LM
from lm_eval.api.instance import Instance
from tokenizers import Tokenizer

from helixdepth.evaluation import encode_pair, score_tokens
from helixdepth.model import LanguageModel
from verify_transfer import sha256


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
    parser.add_argument("--wikitext-revision", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Preserving existing evaluation: {args.output}")
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.manual_seed(2026)
    model = LanguageModel.load(args.model).to(args.device).eval()
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    adapter = HelixHarness(model, tokenizer, args.batch_size)
    started = time.monotonic()
    for task in ("hellaswag", "arc_easy", "piqa", "winogrande"):
        result = lm_eval.simple_evaluate(model=adapter, tasks=[task], num_fewshot=0,
                    limit=args.limit, bootstrap_iters=1000, log_samples=True,
                    random_seed=2026, numpy_random_seed=2026, torch_random_seed=2026, fewshot_random_seed=2026)
        (args.output / f"{task}.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print(task, result["results"], flush=True)
    wiki = wikitext(model, tokenizer, args.wikitext_revision, args.batch_size)
    (args.output / "wikitext103.json").write_text(json.dumps(wiki, indent=2), encoding="utf-8")
    report = {"complete_benchmarks": args.limit is None, "limit": args.limit, "seconds": time.monotonic() - started,
              "model_sha256": sha256(args.model), "tokenizer_sha256": sha256(args.tokenizer),
              "harness_version": importlib.metadata.version("lm_eval"), "torch": str(torch.__version__),
              "batch_size": args.batch_size, "zero_shot": True, "wikitext": wiki,
              "files": {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in args.output.glob("*.json")}}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
