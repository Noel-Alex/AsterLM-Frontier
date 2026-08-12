from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from asterlm.kernels.situ_glu import situ_glu

from .linear import build_linear, mark_residual


class SwiGLU(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        linear_backend: str = "torch",
        *,
        loqt_rank: int = 32,
        loqt_alpha: float = 32.0,
        loqt_group_size: int = 64,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        kwargs = {
            "backend": linear_backend,
            "loqt_rank": loqt_rank,
            "loqt_alpha": loqt_alpha,
            "loqt_group_size": loqt_group_size,
            "init_std": init_std,
        }
        self.gate_up = build_linear(dim, hidden_dim * 2, bias=False, **kwargs)
        self.down = mark_residual(build_linear(hidden_dim, dim, bias=False, **kwargs))
        self.dropout = nn.Dropout(dropout)
        self.activation_name = "swiglu"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.dropout(self.down(F.silu(gate) * up))


class SiTUGLU(nn.Module):
    """Bounded Kimi K3 expert activation, shape-compatible with SwiGLU kernels."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        linear_backend: str = "torch",
        *,
        beta_gate: float = 4.0,
        beta_up: float = 25.0,
        loqt_rank: int = 32,
        loqt_alpha: float = 32.0,
        loqt_group_size: int = 64,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if beta_gate <= 0 or beta_up <= 0:
            raise ValueError("SiTU-GLU beta values must be positive")
        kwargs = {
            "backend": linear_backend,
            "loqt_rank": loqt_rank,
            "loqt_alpha": loqt_alpha,
            "loqt_group_size": loqt_group_size,
            "init_std": init_std,
        }
        self.gate_up = build_linear(dim, hidden_dim * 2, bias=False, **kwargs)
        self.down = mark_residual(build_linear(hidden_dim, dim, bias=False, **kwargs))
        self.dropout = nn.Dropout(dropout)
        self.activation_name = "situ_glu"
        self.beta_gate = float(beta_gate)
        self.beta_up = float(beta_up)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.dropout(
            self.down(situ_glu(gate, up, self.beta_gate, self.beta_up))
        )
