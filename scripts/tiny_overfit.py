"""Deterministic CPU wiring check; these synthetic losses are not language quality."""
import argparse
import json
from pathlib import Path
from typing import Literal

import torch

from helixdepth.config import tiny_config
from helixdepth.model import LanguageModel


def run(variant: Literal["baseline", "helixdepth"], steps: int = 100) -> dict[str, object]:
    torch.manual_seed(2026)
    torch.set_num_threads(1)
    model = LanguageModel(tiny_config(variant))
    tokens = torch.stack([torch.arange(16), (torch.arange(16) + 7) % 32]).long()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
    with torch.no_grad():
        initial = model.loss(tokens).item()
    history = [{"step": 0, "loss": initial}]
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(tokens)
        if not torch.isfinite(loss):
            raise AssertionError("Non-finite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step % 20 == 0 or step == steps:
            with torch.no_grad():
                history.append({"step": step, "loss": model.loss(tokens).item()})
    final = float(history[-1]["loss"])
    passed = final < 0.2 and final < initial * 0.1
    result = {"variant": variant, "seed": 2026, "threads": 1, "device": "cpu",
              "steps": steps, "config": model.config.to_dict(), "optimizer": "AdamW",
              "learning_rate": 0.01, "weight_decay": 0.0, "gradient_clip": 1.0,
              "initial_loss": initial, "final_loss": final, "history": history,
              "threshold": "final < 0.2 and final < 10% of initial", "passed": passed}
    if not passed:
        raise AssertionError(json.dumps(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = json.dumps([run("baseline"), run("helixdepth")], indent=2)
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
