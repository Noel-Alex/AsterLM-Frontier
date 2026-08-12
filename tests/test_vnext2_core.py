from __future__ import annotations

import copy

import torch

from asterlm.config import AsterConfig
from asterlm.layers.latent_attention import LatentAttention
from asterlm.model import AsterLM


def tiny_config(**updates) -> AsterConfig:
    values = dict(
        vocab_size=256,
        d_model=32,
        n_layers=2,
        n_heads=4,
        head_dim=8,
        ffn_hidden=64,
        ffn_type="dense",
        max_seq_len=64,
        linear_backend="torch",
        ffn_linear_backend="torch",
        kda_ratio=0,
        latent_rank=8,
        rope_dim=4,
        attention_window=64,
        sink_tokens=4,
        attention_dropout=0.0,
        attention_gate=False,
        qk_stat_tokens=0,
        mtp_depth=0,
        gradient_checkpointing=False,
        lm_loss_backend="legacy_chunked",
        lm_loss_chunk_size=16,
        attention_train_backend="sdpa",
    )
    values.update(updates)
    return AsterConfig(**values)


def test_absorbed_mla_matches_reconstructed_forward_and_gradients():
    torch.manual_seed(11)
    ref = LatentAttention(tiny_config(attention_train_backend="sdpa"), layer_idx=0)
    absorbed = LatentAttention(tiny_config(attention_train_backend="absorbed_sdpa"), layer_idx=0)
    absorbed.load_state_dict(copy.deepcopy(ref.state_dict()))
    ref.train()
    absorbed.train()

    position_ids = torch.arange(12).unsqueeze(0).expand(2, -1)
    x1 = torch.randn(2, 12, 32, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)

    y1 = ref(x1, position_ids)
    y2 = absorbed(x2, position_ids)
    torch.testing.assert_close(y1, y2, rtol=3e-4, atol=3e-5)

    probe = torch.randn_like(y1)
    (y1 * probe).sum().backward()
    (y2 * probe).sum().backward()
    torch.testing.assert_close(x1.grad, x2.grad, rtol=7e-4, atol=8e-5)
    for name in ("k_up.weight", "v_up.weight", "q_up.weight", "kv_down.weight"):
        p1 = dict(ref.named_parameters())[name]
        p2 = dict(absorbed.named_parameters())[name]
        assert p1.grad is not None and p2.grad is not None
        torch.testing.assert_close(p1.grad, p2.grad, rtol=1e-3, atol=1e-4)


def test_deepseek_mtp_one_step_smoke_and_backward():
    torch.manual_seed(19)
    cfg = tiny_config(
        mtp_depth=1,
        mtp_architecture="deepseek",
        mtp_block_kind="latent",
        mtp_loss_weight=0.1,
        attention_train_backend="absorbed_sdpa",
    )
    model = AsterLM(cfg)
    model.train()
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    labels = torch.roll(ids, shifts=-1, dims=1)
    labels[:, -1] = -100
    labels[:, 7] = -100
    out = model(ids, labels=labels, return_logits=False)
    assert out.main_loss is not None and torch.isfinite(out.main_loss)
    assert out.mtp_loss is not None and torch.isfinite(out.mtp_loss)
    assert out.loss is not None and torch.isfinite(out.loss)
    out.loss.backward()
    assert model.mtp_eh_proj is not None
    assert model.mtp_eh_proj.weight.grad is not None
    assert torch.isfinite(model.mtp_eh_proj.weight.grad).all()



def test_deepseek_mtp_target_shift_and_boundary_mask():
    import types

    torch.manual_seed(23)
    cfg = tiny_config(
        mtp_depth=1,
        mtp_architecture="deepseek",
        mtp_block_kind="latent",
        mtp_loss_weight=0.1,
        attention_train_backend="absorbed_sdpa",
    )
    model = AsterLM(cfg)
    model.train()
    ids = torch.randint(0, cfg.vocab_size, (1, 10))
    labels = torch.roll(ids, shifts=-1, dims=1)
    labels[:, -1] = -100
    # Simulate a packed-document boundary: the ordinary next-token transition at
    # source position 4 is masked, so MTP must not use h_4 to leap to token 6.
    labels[:, 4] = -100

    original = model._projected_cross_entropy
    captured: list[torch.Tensor] = []

    def wrapped(self, hidden, target, ignore_index):
        captured.append(target.detach().clone())
        return original(hidden, target, ignore_index)

    model._projected_cross_entropy = types.MethodType(wrapped, model)
    out = model(ids, labels=labels, return_logits=False)
    assert out.loss is not None and torch.isfinite(out.loss)
    assert len(captured) >= 2
    mtp_target = captured[-1]
    expected = labels[:, 1:].clone()
    expected = expected.masked_fill(labels[:, :-1].eq(-100), -100)
    torch.testing.assert_close(mtp_target, expected)


def test_deepseek_mtp_is_depth_one_research_backend():
    try:
        tiny_config(mtp_depth=2, mtp_architecture="deepseek")
    except ValueError as exc:
        assert "depth 1" in str(exc)
    else:
        raise AssertionError("DeepSeek-style vNext2 MTP must reject depth > 1")


def test_stable_latent_moe_components_are_explicit_and_trainable():
    from asterlm.layers.ffn import SiTUGLU
    from asterlm.layers.latent_moe import LatentMoE
    from asterlm.layers.norm import RMSNorm

    base = tiny_config(
        ffn_type="latent_moe",
        latent_moe_dim=8,
        latent_moe_post_norm=False,
        moe_first_dense_layers=0,
        moe_every=1,
        moe_num_experts=4,
        moe_top_k=2,
        moe_shared_experts=0,
        moe_expert_hidden=16,
    )
    stable = tiny_config(
        ffn_type="latent_moe",
        latent_moe_dim=8,
        latent_moe_post_norm=True,
        moe_activation="situ_glu",
        moe_balance_strategy="quantile",
        moe_first_dense_layers=0,
        moe_every=1,
        moe_num_experts=4,
        moe_top_k=2,
        moe_shared_experts=0,
        moe_expert_hidden=16,
    )
    m0 = AsterLM(base)
    m1 = AsterLM(stable)
    f0 = m0.blocks[0].ffn
    f1 = m1.blocks[0].ffn
    assert isinstance(f0, LatentMoE) and isinstance(f1, LatentMoE)
    assert isinstance(f0.routed_post_norm, torch.nn.Identity)
    assert isinstance(f1.routed_post_norm, RMSNorm)
    assert isinstance(f1.routed[0], SiTUGLU)
    assert f1.balance_strategy == "quantile"
    ids = torch.randint(0, stable.vocab_size, (2, 12))
    labels = torch.roll(ids, shifts=-1, dims=1)
    labels[:, -1] = -100
    out = m1(ids, labels=labels, return_logits=False)
    assert out.loss is not None and torch.isfinite(out.loss)
    out.loss.backward()
    assert f1.routed_post_norm.weight.grad is not None
    assert torch.isfinite(f1.routed_post_norm.weight.grad).all()
    old_bias = f1.routing_bias.clone()
    histogram_load = f1.update_routing_bias()
    assert histogram_load is not None
    assert torch.isfinite(f1.routing_bias).all()
    assert torch.count_nonzero(f1.quantile_histogram) == 0
    assert not torch.equal(old_bias, f1.routing_bias) or torch.allclose(
        f1.routing_bias, torch.zeros_like(f1.routing_bias)
    )


def test_situ_glu_is_locally_swiglu_like_and_globally_bounded():
    from asterlm.layers.ffn import situ_glu

    small_gate = torch.tensor([-0.01, 0.01])
    small_up = torch.tensor([0.02, -0.02])
    expected = torch.nn.functional.silu(small_gate) * small_up
    torch.testing.assert_close(
        situ_glu(small_gate, small_up), expected, atol=2e-7, rtol=2e-4
    )
    huge = situ_glu(torch.tensor([1000.0]), torch.tensor([1000.0]))
    assert 0.0 < float(huge) <= 100.0
