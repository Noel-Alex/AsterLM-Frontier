from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Conventional per-channel RMSNorm."""

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        y = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        if self.weight is not None:
            y = y * self.weight.float()
        return y.to(dtype)


class SingleScaleRMSNorm(nn.Module):
    """Outlier-Safe Pre-Training's scalar-affine RMSNorm.

    The normalization statistics are unchanged, but every channel shares one learned
    scale instead of receiving an independent scale. This removes a direct source of
    channel-wise amplification while preserving a learnable global gain.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        del dim  # Kept in the signature so it is interchangeable with RMSNorm.
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        y = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).to(dtype)


def build_norm(dim: int, eps: float, norm_type: str = "rmsnorm") -> nn.Module:
    if norm_type == "rmsnorm":
        return RMSNorm(dim, eps)
    if norm_type == "ssnorm":
        return SingleScaleRMSNorm(dim, eps)
    raise ValueError(f"Unsupported norm_type={norm_type!r}")


class HeadRMSNorm(nn.Module):
    """RMS normalization over the last (per-head) dimension.

    This is used inside the KDA recurrence/output operator and intentionally remains
    per-head even when transformer block norms use SSNorm.
    """

    def __init__(self, head_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        y = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).to(dtype)
