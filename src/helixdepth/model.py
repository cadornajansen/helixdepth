import math
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import ModelConfig
from .hypernetwork import DepthConditioner, LowRankBasis, PROJECTIONS


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight


def apply_rope(x: Tensor, theta: float) -> Tensor:
    """Rotate adjacent feature pairs; positions start at zero (no KV cache)."""
    head_width = x.size(-1)
    frequencies = theta ** (-torch.arange(0, head_width, 2, device=x.device, dtype=torch.float32) / head_width)
    angles = torch.arange(x.size(-2), device=x.device, dtype=torch.float32)[:, None] * frequencies
    cosine, sine = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1).flatten(-2)


def projection_shapes(config: ModelConfig) -> dict[str, tuple[int, int]]:
    d, f = config.width, config.ffn_width
    return {"q": (d, d), "k": (d, d), "v": (d, d), "out": (d, d),
            "gate": (d, f), "up": (d, f), "down": (f, d)}


class BaseBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.projections = nn.ModuleDict({
            name: nn.Linear(i, o, bias=False) for name, (i, o) in projection_shapes(config).items()
        })
        for name, layer in self.projections.items():
            std = 0.02 / math.sqrt(2 * config.stages) if name in ("out", "down") else 0.02
            nn.init.normal_(layer.weight, std=std)

    def forward(self, x: Tensor, norms: nn.ModuleList, bases: nn.ModuleDict,
                coefficients: Tensor | None) -> Tensor:
        def project(name: str, value: Tensor) -> Tensor:
            result = self.projections[name](value)
            if coefficients is not None:
                result = result + bases[name](value, coefficients[PROJECTIONS.index(name)])
            return result

        normalized = norms[0](x)
        batch, length, width = x.shape
        q, k, v = [project(name, normalized).reshape(batch, length, self.config.heads, -1).transpose(1, 2)
                   for name in ("q", "k", "v")]
        q, k = apply_rope(q, self.config.rope_theta), apply_rope(k, self.config.rope_theta)
        attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        x = x + project("out", attended.transpose(1, 2).reshape(batch, length, width))
        normalized = norms[1](x)
        return x + project("down", F.silu(project("gate", normalized)) * project("up", normalized))


class LanguageModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.schedule = tuple(stage % config.base_blocks for stage in range(config.stages))
        self.embedding = nn.Embedding(config.vocab_size, config.width)
        nn.init.normal_(self.embedding.weight, std=0.02)
        self.blocks = nn.ModuleList([BaseBlock(config) for _ in range(config.base_blocks)])
        self.stage_norms = nn.ModuleList([
            nn.ModuleList([RMSNorm(config.width, config.norm_eps), RMSNorm(config.width, config.norm_eps)])
            for _ in range(config.stages)
        ])
        self.final_norm = RMSNorm(config.width, config.norm_eps)
        self.head = nn.Linear(config.width, config.vocab_size, bias=False)
        self.head.weight = self.embedding.weight
        self.bases = nn.ModuleDict()
        self.conditioner: DepthConditioner | None = None
        if config.variant == "helixdepth":
            self.bases.update({name: LowRankBasis(i, o, config.rank)
                               for name, (i, o) in projection_shapes(config).items()})
            self.conditioner = DepthConditioner(config)

    def forward(self, input_ids: Tensor) -> Tensor:
        if input_ids.ndim != 2 or not 1 <= input_ids.size(1) <= self.config.max_context:
            raise ValueError("Expected [batch, sequence] with sequence within max_context")
        x = self.embedding(input_ids)
        coefficients = self.conditioner() if self.conditioner is not None else None
        for stage, block_index in enumerate(self.schedule):
            x = self.blocks[block_index](x, self.stage_norms[stage], self.bases,
                                         None if coefficients is None else coefficients[stage])
        return self.head(self.final_norm(x))

    def loss(self, input_ids: Tensor) -> Tensor:
        if input_ids.ndim != 2 or input_ids.size(1) < 2:
            raise ValueError("Next-token loss needs at least two tokens")
        logits = self(input_ids)
        return F.cross_entropy(logits[:, :-1].reshape(-1, self.config.vocab_size),
                               input_ids[:, 1:].reshape(-1))

    def save(self, path: str | Path) -> None:
        torch.save({"config": self.config.to_dict(), "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str | Path) -> "LanguageModel":
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(ModelConfig(**checkpoint["config"]))
        model.load_state_dict(checkpoint["state_dict"])
        return model
