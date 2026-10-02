"""Generate bounded text continuations from our saved models, without training."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any

import torch
from tokenizers import Tokenizer

from helixdepth.model import LanguageModel
from helixdepth.packing import sha256_file


PROMPTS = (
    "Plants need sunlight because",
    "When rain falls on a mountain,",
    "To solve a difficult problem, it helps to",
)


def generate(model: LanguageModel, prompt_ids: list[int], max_new_tokens: int,
             temperature: float, top_k: int, seed: int,
             blocked_ids: list[int], eos_id: int) -> dict[str, Any]:
    if not prompt_ids or not 1 <= max_new_tokens <= 256:
        raise ValueError("Use a nonempty prompt and 1 to 256 new tokens")
    if len(prompt_ids) + max_new_tokens > model.config.max_context:
        raise ValueError("Prompt plus requested output must fit the model context")
    if not math.isfinite(temperature) or temperature < 0 or not 1 <= top_k <= model.config.vocab_size:
        raise ValueError("Temperature must be finite/nonnegative; top-k must fit the vocabulary")
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    generated: list[int] = []
    stop_reason = "max_new_tokens"
    prior_mode = model.training
    model.eval()
    try:
        with torch.inference_mode():
            # One untimed forward pass warms up this prompt's execution path.
            model(torch.tensor([prompt_ids], dtype=torch.long, device=device))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            for _ in range(max_new_tokens):
                inputs = torch.tensor([prompt_ids + generated], dtype=torch.long, device=device)
                logits = model(inputs)[0, -1].float()
                if not torch.isfinite(logits).all():
                    raise ValueError("Non-finite generation logits")
                logits[blocked_ids] = -torch.inf
                if temperature == 0:
                    next_id = int(logits.argmax().item())
                else:
                    values, indices = torch.topk(logits / temperature, top_k)
                    choice = torch.multinomial(values.softmax(-1), 1, generator=generator)
                    next_id = int(indices[choice].item())
                generated.append(next_id)
                if next_id == eos_id:
                    stop_reason = "eos"
                    break
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
    finally:
        model.train(prior_mode)
    return {"generated_token_ids": generated, "generated_tokens_including_eos": len(generated),
            "stop_reason": stop_reason, "seconds": elapsed,
            "tokens_per_second": len(generated) / elapsed, "seed": seed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New JSON file; existing files are preserved")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--prompt", action="append", help="Repeat for multiple prompts; defaults to three fixed examples")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.8, help="Zero selects greedy decoding")
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Preserving existing demo: {args.output}")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    model_hash = sha256_file(args.model)
    tokenizer_hash = sha256_file(args.tokenizer)
    model = LanguageModel.load(args.model).to(args.device).eval()
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    if tokenizer.get_vocab_size() != model.config.vocab_size:
        raise ValueError("Tokenizer vocabulary does not match the model")
    tokenizer_data = json.loads(args.tokenizer.read_text(encoding="utf-8"))
    eos_id = 2  # Same document boundary token as the frozen training corpus.
    blocked = [row["id"] for row in tokenizer_data["added_tokens"]
               if row["special"] and row["id"] != eos_id]
    results = []
    for index, prompt in enumerate(args.prompt or PROMPTS):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
        row = generate(model, prompt_ids, args.max_new_tokens, args.temperature,
                       args.top_k, args.seed + index, blocked, eos_id)
        row.update({"prompt": prompt, "prompt_token_ids": prompt_ids,
                    "continuation": tokenizer.decode(row["generated_token_ids"], skip_special_tokens=True),
                    "full_text": tokenizer.decode(prompt_ids + row["generated_token_ids"], skip_special_tokens=True)})
        results.append(row)
        print(f"\n[{model.config.variant}] Prompt: {prompt}\n{row['continuation']}\n"
              f"{row['generated_tokens_including_eos']} tokens; {row['seconds']:.3f}s; "
              f"{row['tokens_per_second']:.2f} tokens/s; stop={row['stop_reason']}", flush=True)
    if sha256_file(args.model) != model_hash or sha256_file(args.tokenizer) != tokenizer_hash:
        raise ValueError("Input files changed during generation")
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
              "variant": model.config.variant, "model_sha256": model_hash,
              "tokenizer_sha256": tokenizer_hash, "script_sha256": sha256_file(Path(__file__)),
              "torch": str(torch.__version__), "device": args.device,
              "hardware": torch.cuda.get_device_name() if args.device == "cuda" else "CPU",
              "config": model.config.to_dict(), "precision": "float32",
              "decoding": {"max_new_tokens": args.max_new_tokens, "temperature": args.temperature,
                           "top_k": args.top_k, "seed": args.seed, "blocked_special_ids": blocked,
                           "eos_id": eos_id, "kv_cache": False},
              "timing_scope": "Warm generation loop including host sampling/transfer and repeated context forwards; excludes model loading, warmup, text encoding/decoding. CUDA synchronized. Illustrative single-run timing, not a throughput benchmark.",
              "note": "Base language-model continuations, not an instruction-tuned assistant or evidence of reasoning accuracy. Seeded outputs can vary across devices/library versions.",
              "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(f"Saved all {len(results)} examples: {args.output}", flush=True)


if __name__ == "__main__":
    main()
