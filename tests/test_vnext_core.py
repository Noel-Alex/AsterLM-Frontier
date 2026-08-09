from __future__ import annotations

import math

import pytest
import torch

from asterlm.config import AsterConfig
from asterlm.layers.attnres_vnext import AttnResMix
from asterlm.layers.latent_moe import LatentMoE
from asterlm.model import AsterLM


def tiny_config(**overrides) -> AsterConfig:
    values = dict(
        vocab_size=256,
        d_model=64,
        n_layers=2,
        n_heads=4,
        head_dim=16,
        ffn_hidden=128,
        ffn_type="dense",
        kda_ratio=0,
        latent_rank=16,
        q_lora_rank=32,
        rope_dim=8,
        max_seq_len=128,
        attention_window=128,
        sink_tokens=4,
        embedding_projection=False,
        norm_type="rmsnorm",
        mtp_depth=0,
        lm_loss_chunk_size=16,
        gradient_checkpointing=True,
        linear_backend="torch",
    )
    values.update(overrides)
    return AsterConfig(**values)


def test_activation_checkpoint_is_not_discarded():
    """Regression test for the critical double-forward checkpoint bug."""
    torch.manual_seed(7)
    model = AsterLM(tiny_config())
    model.train()
    counts = [0 for _ in model.blocks]
    handles = []
    for i, block in enumerate(model.blocks):
        def hook(_module, _inputs, _output, idx=i):
            counts[idx] += 1
        handles.append(block.register_forward_hook(hook))

    ids = torch.randint(0, model.config.vocab_size, (1, 32))
    labels = torch.randint(0, model.config.vocab_size, (1, 32))
    out = model(ids, labels=labels, return_logits=False)
    # A checkpointed layer runs exactly once in the initial forward. The old bug
    # ran every block twice before backward because the checkpoint result was dropped.
    assert counts == [1] * len(model.blocks)
    assert out.loss is not None and torch.isfinite(out.loss)
    out.loss.backward()
    assert all(1 <= c <= 2 for c in counts)
    for h in handles:
        h.remove()


def test_linear_cross_entropy_matches_legacy_fp32():
    if not hasattr(torch.nn.functional, "linear_cross_entropy"):
        pytest.skip("PyTorch LinearCrossEntropy is unavailable")
    torch.manual_seed(11)
    model = AsterLM(tiny_config(gradient_checkpointing=False))
    model.train()
    h0 = torch.randn(2, 24, model.config.d_model, requires_grad=True)
    labels = torch.randint(0, model.config.vocab_size, (2, 24))
    labels[0, 0] = -100

    model.config.lm_loss_backend = "legacy_chunked"
    legacy = model._projected_cross_entropy(h0, labels, -100)
    legacy.backward()
    g_hidden_legacy = h0.grad.detach().clone()
    g_weight_legacy = model.lm_head.weight.grad.detach().clone()

    model.zero_grad(set_to_none=True)
    h1 = h0.detach().clone().requires_grad_(True)
    model.config.lm_loss_backend = "torch_linear_ce"
    modern = model._projected_cross_entropy(h1, labels, -100)
    modern.backward()

    assert torch.allclose(legacy.detach(), modern.detach(), atol=2e-5, rtol=2e-5)
    assert torch.allclose(g_hidden_legacy, h1.grad, atol=2e-4, rtol=2e-4)
    assert torch.allclose(g_weight_legacy, model.lm_head.weight.grad, atol=3e-4, rtol=3e-4)


def test_latent_moe_reference_forward_backward():
    torch.manual_seed(13)
    layer = LatentMoE(
        dim=64,
        latent_dim=16,
        expert_hidden=96,
        num_experts=8,
        top_k=2,
        shared_experts=1,
        linear_backend="torch",
        moe_impl="reference",
    )
    x = torch.randn(2, 17, 64, requires_grad=True)
    y = layer(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    y.float().square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert layer.last_load is not None
    assert math.isclose(float(layer.last_load.sum()), 1.0, rel_tol=0.0, abs_tol=1e-5)


def test_attnres_fallback_has_finite_gradients():
    torch.manual_seed(17)
    mix = AttnResMix(32)
    residuals = [torch.randn(2, 7, 32, requires_grad=True) for _ in range(4)]
    out = mix._torch_forward(residuals)
    assert out.shape == residuals[0].shape
    assert torch.isfinite(out).all()
    out.square().mean().backward()
    assert all(r.grad is not None and torch.isfinite(r.grad).all() for r in residuals)
    # Zero query starts as a uniform depth average.
    expected = torch.stack([r.detach() for r in residuals]).mean(0)
    assert torch.allclose(out.detach(), expected, atol=1e-6, rtol=1e-6)


def test_vnext_config_accepts_gdn2_and_latent_moe():
    cfg = tiny_config(
        n_layers=4,
        layer_pattern=["gdn2", "gdn2", "gdn2", "latent"],
        ffn_type="latent_moe",
        latent_moe_dim=16,
        moe_num_experts=16,
        moe_top_k=4,
        moe_expert_hidden=96,
    )
    assert cfg.pattern.count("gdn2") == 3
    assert cfg.ffn_type == "latent_moe"


def test_attnres_integrated_checkpoint_forward_backward():
    """Exercise the tuple-valued AttnRes checkpoint path, not only the mixer op."""
    torch.manual_seed(23)
    model = AsterLM(
        tiny_config(
            n_layers=4,
            use_block_attnres=True,
            attnres_block_size=4,
            gradient_checkpointing=True,
        )
    )
    model.train()
    ids = torch.randint(0, model.config.vocab_size, (1, 24))
    labels = torch.randint(0, model.config.vocab_size, (1, 24))
    out = model(ids, labels=labels, return_logits=False)
    assert out.loss is not None and torch.isfinite(out.loss)
    out.loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)


def test_segment_checkpoint_saves_only_segment_boundaries_without_double_forward():
    """Segmented checkpointing must preserve one initial forward per block."""
    torch.manual_seed(29)
    model = AsterLM(
        tiny_config(
            n_layers=4,
            checkpoint_segment_size=2,
            gradient_checkpointing=True,
        )
    )
    model.train()
    counts = [0 for _ in model.blocks]
    handles = []
    for i, block in enumerate(model.blocks):
        def hook(_module, _inputs, _output, idx=i):
            counts[idx] += 1
        handles.append(block.register_forward_hook(hook))

    ids = torch.randint(0, model.config.vocab_size, (1, 32))
    labels = torch.randint(0, model.config.vocab_size, (1, 32))
    out = model(ids, labels=labels, return_logits=False)
    assert counts == [1] * len(model.blocks)
    assert out.loss is not None and torch.isfinite(out.loss)
    out.loss.backward()
    assert all(1 <= c <= 2 for c in counts)
    for h in handles:
        h.remove()
