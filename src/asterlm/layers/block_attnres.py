from __future__ import annotations

import math

import torch
from torch import nn

from .norm import build_norm


class DepthResidualMixer(nn.Module):
    """Token-wise attention over earlier block-boundary residual streams.

    This is a compact Block-AttnRes-inspired experimental module. It is disabled in
    the laptop default because storing several depth streams increases activation
    memory. Enable it only for controlled ablations.
    """

    def __init__(
        self, dim: int, key_dim: int, max_states: int, eps: float = 1e-6, norm_type: str = "rmsnorm"
    ) -> None:
        super().__init__()
        self.max_states = max_states
        self.norm = build_norm(dim, eps, norm_type)
        self.key = nn.Linear(dim, key_dim, bias=False)
        self.query = nn.Parameter(torch.empty(max_states, key_dim))
        nn.init.normal_(self.query, std=key_dim**-0.5)

    def forward(self, states: list[torch.Tensor]) -> torch.Tensor:
        if not states:
            raise ValueError("DepthResidualMixer needs at least one state")
        states = states[-self.max_states :]
        stack = torch.stack([self.norm(x) for x in states], dim=2)  # B,T,N,D
        keys = self.key(stack)  # B,T,N,K
        n = len(states)
        q = self.query[:n]
        scores = torch.einsum("btnk,nk->btn", keys, q) / math.sqrt(keys.shape[-1])
        weights = scores.softmax(dim=-1).unsqueeze(-1)
        return (weights * stack).sum(dim=2)
