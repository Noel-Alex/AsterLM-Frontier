from __future__ import annotations

import torch

from asterlm.layers.moe import DeepSeekStyleMoE


def test_reference_switch_keeps_cpu_path_functional(monkeypatch):
    monkeypatch.setenv("ASTER_MOE_IMPL", "reference")
    moe = DeepSeekStyleMoE(
        dim=32,
        expert_hidden=48,
        num_experts=4,
        top_k=2,
        shared_experts=1,
        linear_backend="torch",
    )
    x = torch.randn(2, 7, 32, requires_grad=True)
    y = moe(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    y.square().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_grouped_switch_is_explicit(monkeypatch):
    monkeypatch.setenv("ASTER_MOE_IMPL", "grouped")
    try:
        DeepSeekStyleMoE(
            dim=32,
            expert_hidden=48,
            num_experts=4,
            top_k=2,
            shared_experts=1,
            linear_backend="torch",
        )
    except ValueError as exc:
        assert "requires linear_backend='transformer_engine'" in str(exc)
    else:
        raise AssertionError("grouped mode should reject the torch backend")


def test_reference_state_dict_has_no_grouped_bridge_keys(monkeypatch):
    monkeypatch.setenv("ASTER_MOE_IMPL", "reference")
    moe = DeepSeekStyleMoE(
        dim=32,
        expert_hidden=48,
        num_experts=4,
        top_k=2,
        shared_experts=1,
        linear_backend="torch",
    )
    keys = set(moe.state_dict())
    assert any(k.startswith("routed.0.") for k in keys)
    assert not any("grouped" in k for k in keys)


def test_grouped_dispatch_switch_rejects_unknown_value(monkeypatch):
    monkeypatch.setenv("ASTER_MOE_IMPL", "grouped")
    monkeypatch.setenv("ASTER_MOE_DISPATCH", "unknown")
    try:
        DeepSeekStyleMoE(
            dim=32,
            expert_hidden=48,
            num_experts=4,
            top_k=2,
            shared_experts=1,
            linear_backend="transformer_engine",
        )
    except ValueError as exc:
        assert "ASTER_MOE_DISPATCH" in str(exc)
    else:
        raise AssertionError("unknown grouped dispatch should be rejected")
