"""Count actual registered parameters, using metadata-only model allocation."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from helixdepth.config import ModelConfig
from helixdepth.model import LanguageModel

ROOT = Path(__file__).resolve().parent


def count_model(config: ModelConfig) -> dict[str, object]:
    with torch.device("meta"):
        model = LanguageModel(config)
    assert model.head.weight is model.embedding.weight
    assert len({id(block) for block in model.blocks}) == config.base_blocks
    assert model.schedule == tuple(i % config.base_blocks for i in range(config.stages))
    breakdown: dict[str, int] = defaultdict(int)
    for name, parameter in model.named_parameters():
        group = ".".join(name.split(".")[:2]) if name.startswith("conditioner.") else name.split(".")[0]
        breakdown[group] += parameter.numel()
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    raw = sum(p.numel() for _, p in model.named_parameters(remove_duplicate=False))
    assert raw - total == config.vocab_size * config.width, "Only head/embedding should alias"
    d, f = config.width, config.ffn_width
    analytical = config.vocab_size * d + config.base_blocks * (4*d*d + 3*d*f) + 2*d*config.stages + d
    if config.variant == "helixdepth":
        analytical += config.rank * (8*d + 3*(d+f))
        analytical += config.stages*config.depth_width
        analytical += (config.depth_width+1)*config.hyper_width + (config.hyper_width+1)*7*config.rank
    assert total == analytical, (total, analytical)
    assert total <= 50_000_000 and trainable <= 50_000_000
    return {"variant": config.variant, "total": total, "trainable": trainable,
            "analytical": analytical, "breakdown": dict(breakdown),
            "tied_parameters_excluded": raw-total, "stored_blocks": len(model.blocks),
            "schedule": list(model.schedule), "device": "meta"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = [count_model(ModelConfig.from_yaml(ROOT / "configs" / f"{name}.yaml"))
               for name in ("baseline", "helixdepth")]
    assert [result["total"] for result in results] == [40_968_320, 41_184_432]
    output = json.dumps(results, indent=2)
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
