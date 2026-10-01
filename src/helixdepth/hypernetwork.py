import math

import torch
from torch import Tensor, nn

from .config import ModelConfig

PROJECTIONS = ("q", "k", "v", "out", "gate", "up", "down")


class LowRankBasis(nn.Module):
    def __init__(self, in_width: int, out_width: int, rank: int) -> None:
        super().__init__()
        self.A = nn.Parameter(torch.empty(in_width, rank))
        self.B = nn.Parameter(torch.empty(rank, out_width))
        nn.init.normal_(self.A, std=1 / math.sqrt(in_width))
        nn.init.normal_(self.B, std=0.02 / math.sqrt(rank))

    def forward(self, x: Tensor, coefficients: Tensor) -> Tensor:
        return ((x @ self.A) * coefficients) @ self.B


class DepthConditioner(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.rank = config.rank
        self.depth = nn.Embedding(config.stages, config.depth_width)
        self.network = nn.Sequential(
            nn.Linear(config.depth_width, config.hyper_width), nn.SiLU(),
            nn.Linear(config.hyper_width, len(PROJECTIONS) * config.rank),
        )
        nn.init.normal_(self.depth.weight, std=0.02)
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                nn.init.zeros_(layer.bias)

    def forward(self) -> Tensor:
        return self.network(self.depth.weight).reshape(-1, len(PROJECTIONS), self.rank)
