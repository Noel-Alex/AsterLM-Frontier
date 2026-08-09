from __future__ import annotations

import torch
from torch import nn

from asterlm.layers.moe import DeepSeekStyleMoE


class AlignmentCheckingExpert(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[0] % 16 == 0
        return x * 2


def _moe() -> DeepSeekStyleMoE:
    model = DeepSeekStyleMoE(
        dim=32,
        expert_hidden=32,
        num_experts=2,
        top_k=1,
        shared_experts=0,
        linear_backend="torch",
    )
    model.linear_backend = "transformer_engine"
    return model


def test_te_expert_input_is_padded_and_output_is_sliced():
    model = _moe()
    x = torch.randn(17, 32)
    out = model._run_expert(AlignmentCheckingExpert(), x)
    assert out.shape == x.shape
    torch.testing.assert_close(out, x * 2)


def test_te_expert_aligned_input_is_unchanged_in_shape():
    model = _moe()
    x = torch.randn(32, 32)
    out = model._run_expert(AlignmentCheckingExpert(), x)
    assert out.shape == x.shape
    torch.testing.assert_close(out, x * 2)
