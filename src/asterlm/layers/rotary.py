from __future__ import annotations

import math

import torch
from torch import nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """RoPE with optional linear or YaRN-style frequency interpolation.

    The YaRN implementation follows the public interpolation/ramp formulation and is
    intentionally kept in Python so experiments remain inspectable.
    """

    def __init__(
        self,
        dim: int,
        theta: float = 1_000_000.0,
        max_position: int = 8192,
        scaling_type: str = "none",
        scaling_factor: float = 1.0,
        original_max_position: int = 8192,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.max_position = max_position
        self.scaling_type = scaling_type
        self.scaling_factor = scaling_factor
        self.original_max_position = original_max_position
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow

        idx = torch.arange(0, dim, 2, dtype=torch.float32)
        base_inv_freq = 1.0 / (theta ** (idx / dim))
        inv_freq = self._scaled_frequencies(base_inv_freq)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _scaled_frequencies(self, inv_freq: torch.Tensor) -> torch.Tensor:
        if self.scaling_type == "none" or self.scaling_factor == 1.0:
            return inv_freq
        if self.scaling_type == "linear":
            return inv_freq / self.scaling_factor
        if self.scaling_type != "yarn":
            raise ValueError(f"Unknown RoPE scaling type: {self.scaling_type}")

        # Number of rotations at the original context length for each frequency.
        rotations = self.original_max_position * inv_freq / (2 * math.pi)
        low = self.beta_slow
        high = self.beta_fast
        ramp = ((rotations - low) / max(high - low, 1e-6)).clamp(0, 1)
        interpolated = inv_freq / self.scaling_factor
        return interpolated * (1 - ramp) + inv_freq * ramp

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # position_ids: [B, T] or [T]
        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)
        freqs = torch.einsum("bt,d->btd", position_ids.float(), self.inv_freq.float())
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


def apply_rotary(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    # x [B,T,H,D], cos/sin [B,T,D]
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    return x * cos + rotate_half(x) * sin
