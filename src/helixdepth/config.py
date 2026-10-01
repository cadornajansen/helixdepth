from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import yaml


@dataclass(frozen=True)
class ModelConfig:
    variant: Literal["baseline", "helixdepth"] = "baseline"
    vocab_size: int = 16384
    max_context: int = 512
    width: int = 640
    heads: int = 10
    ffn_width: int = 1792
    base_blocks: int = 6
    stages: int = 6
    rank: int = 16
    depth_width: int = 32
    hyper_width: int = 64
    norm_eps: float = 1e-6
    rope_theta: float = 10000.0

    def __post_init__(self) -> None:
        if self.variant not in ("baseline", "helixdepth"):
            raise ValueError("Unknown model variant")
        for name in ("vocab_size", "max_context", "width", "heads", "ffn_width",
                     "base_blocks", "stages", "rank", "depth_width", "hyper_width"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.width % self.heads or (self.width // self.heads) % 2:
            raise ValueError("Head width must be an even integer for RoPE")
        if self.stages < self.base_blocks:
            raise ValueError("Every stored block must be used")
        if self.variant == "baseline" and self.stages != self.base_blocks:
            raise ValueError("Baseline uses each independent block exactly once")
        if self.norm_eps <= 0 or self.rope_theta <= 0:
            raise ValueError("Numerical constants must be positive")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelConfig":
        return cls(**yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)


def tiny_config(variant: Literal["baseline", "helixdepth"]) -> ModelConfig:
    return ModelConfig(variant=variant, vocab_size=32, max_context=16, width=32,
                       heads=4, ffn_width=64, base_blocks=2,
                       stages=2 if variant == "baseline" else 4,
                       rank=4, depth_width=8, hyper_width=16)
