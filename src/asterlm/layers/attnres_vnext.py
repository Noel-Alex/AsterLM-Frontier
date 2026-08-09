from __future__ import annotations

import torch
from torch import nn


class AttnResMix(nn.Module):
    """Block Attention-Residual depth mixer.

    This follows the core Attention Residuals operation: normalize each candidate
    residual stream for key construction, score the depth candidates with a learned
    pseudo-query, softmax across depth, and use those weights to mix the *unnormalized*
    residual values. The pseudo-query is zero-initialized so the model begins with an
    equal-weight average over the available depth states.

    On CUDA, current flash-linear-attention builds provide a fused AttnRes kernel. A
    mathematically equivalent PyTorch fallback is kept for CPU tests and capability
    fallbacks. Aster applies its ordinary block norm after this depth mixer, so SSNorm
    remains independent of the AttnRes key RMSNorm.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.query = nn.Parameter(torch.zeros(self.dim))
        self.rms_weight = nn.Parameter(torch.ones(self.dim))

    @staticmethod
    def fused_available() -> bool:
        try:
            from fla.ops.attnres import fused_attnres  # noqa: F401

            return True
        except Exception:
            return False

    def _torch_forward(self, residuals: list[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack(residuals, dim=0)  # [L,B,T,D]
        key = stacked.float()
        key = key * torch.rsqrt(key.square().mean(dim=-1, keepdim=True) + self.eps)
        key = key * self.rms_weight.float()
        scores = torch.einsum("d,lbtd->lbt", self.query.float(), key)
        weights = scores.softmax(dim=0).to(stacked.dtype).unsqueeze(-1)
        return (stacked * weights).sum(dim=0)

    def forward(self, residuals: list[torch.Tensor]) -> torch.Tensor:
        if not residuals:
            raise ValueError("AttnRes requires at least one residual state")
        if len(residuals) == 1:
            return residuals[0]
        shape = residuals[0].shape
        if any(t.shape != shape for t in residuals):
            raise ValueError("All AttnRes residual states must have the same shape")

        if residuals[0].is_cuda and self.fused_available():
            from fla.ops.attnres import fused_attnres

            return fused_attnres(
                query=self.query,
                residuals=residuals,
                rms_weight=self.rms_weight,
                rms_eps=self.eps,
            )
        return self._torch_forward(residuals)
