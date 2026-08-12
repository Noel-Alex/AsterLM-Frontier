from __future__ import annotations

import pytest
import torch

from asterlm.layers.latent_moe import LatentMoE
from asterlm.layers.moe import DeepSeekStyleMoE


def test_reference_switch_keeps_cpu_path_functional():
    moe = DeepSeekStyleMoE(
        dim=32,
        expert_hidden=48,
        num_experts=4,
        top_k=2,
        shared_experts=1,
        linear_backend="torch",
        moe_impl="reference",
    )
    x = torch.randn(2, 7, 32, requires_grad=True)
    y = moe(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    y.square().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_grouped_switch_is_explicit():
    try:
        DeepSeekStyleMoE(
            dim=32,
            expert_hidden=48,
            num_experts=4,
            top_k=2,
            shared_experts=1,
            linear_backend="torch",
            moe_impl="grouped",
        )
    except ValueError as exc:
        assert "requires linear_backend='transformer_engine'" in str(exc)
    else:
        raise AssertionError("grouped mode should reject the torch backend")


def test_reference_state_dict_has_no_grouped_bridge_keys():
    moe = DeepSeekStyleMoE(
        dim=32,
        expert_hidden=48,
        num_experts=4,
        top_k=2,
        shared_experts=1,
        linear_backend="torch",
        moe_impl="reference",
    )
    keys = set(moe.state_dict())
    assert any(k.startswith("routed.0.") for k in keys)
    assert not any("grouped" in k for k in keys)


def test_liger_refuses_to_approximate_k3_situ_activation():
    with pytest.raises(ValueError, match="supports SwiGLU only"):
        LatentMoE(
            dim=32,
            latent_dim=16,
            expert_hidden=32,
            num_experts=4,
            top_k=2,
            shared_experts=0,
            moe_impl="liger",
            activation="situ_glu",
        )
