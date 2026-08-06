from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .linear import build_linear, mark_residual
from .norm import build_norm


class MultiTokenHead(nn.Module):
    """Low-rank future-token head used for auxiliary MTP and self-speculation."""

    def __init__(
        self, dim: int, rank: int, eps: float, linear_backend: str = "torch", norm_type: str = "rmsnorm"
    ) -> None:
        super().__init__()
        self.norm = build_norm(dim, eps, norm_type)
        self.up = build_linear(dim, rank * 2, bias=False, backend=linear_backend)
        self.down = mark_residual(build_linear(rank, dim, bias=False, backend=linear_backend))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        gate, value = self.up(self.norm(hidden)).chunk(2, dim=-1)
        return hidden + self.down(F.silu(gate) * value)


class MultiTokenPredictor(nn.Module):
    def __init__(
        self,
        dim: int,
        rank: int,
        depth: int,
        eps: float,
        linear_backend: str = "torch",
        norm_type: str = "rmsnorm",
    ) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            [MultiTokenHead(dim, rank, eps, linear_backend, norm_type) for _ in range(depth)]
        )

    def forward(self, hidden: torch.Tensor) -> list[torch.Tensor]:
        outputs: list[torch.Tensor] = []
        x = hidden
        for head in self.heads:
            x = head(x)
            outputs.append(x)
        return outputs
