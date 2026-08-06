import torch

from asterlm import AsterConfig, AsterLM, TrainConfig
from asterlm.layers.moe import DeepSeekStyleMoE
from asterlm.optim import build_hybrid_optimizer


def tiny_moe_config() -> AsterConfig:
    return AsterConfig(
        vocab_size=96,
        d_model=32,
        n_layers=4,
        n_heads=2,
        head_dim=16,
        ffn_hidden=96,
        ffn_type="moe",
        moe_first_dense_layers=1,
        moe_num_experts=4,
        moe_top_k=2,
        moe_shared_experts=1,
        moe_expert_hidden=32,
        moe_balance_strategy="hybrid",
        max_seq_len=32,
        kda_ratio=1,
        kda_backend="torch",
        latent_rank=8,
        rope_dim=8,
        attention_window=None,
        sink_tokens=0,
        mtp_depth=1,
        mtp_rank=16,
        gradient_checkpointing=False,
    )


def test_moe_forward_backward_and_active_count():
    model = AsterLM(tiny_moe_config())
    ids = torch.randint(0, 96, (2, 12))
    output = model(ids, labels=ids, return_logits=False)
    assert output.router_aux_loss is not None
    assert output.router_z_loss is not None
    assert output.expert_load is not None
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert model.active_parameter_count() < model.parameter_count()


def test_aux_free_bias_update_moves_away_from_overloaded_expert():
    moe = DeepSeekStyleMoE(16, 24, 4, 2, balance_strategy="bias", bias_update_speed=0.01)
    moe.load_accumulator.copy_(torch.tensor([0.8, 0.1, 0.05, 0.05]))
    moe.load_batches.fill_(1)
    load = moe.update_routing_bias()
    assert load is not None
    assert moe.routing_bias[0] < 0
    assert moe.routing_bias[2] > 0
    assert torch.isclose(moe.routing_bias.mean(), torch.tensor(0.0))


def test_router_stays_out_of_muon_partition():
    model = AsterLM(tiny_moe_config())
    optimizer = build_hybrid_optimizer(model, TrainConfig(device="cpu", max_steps=2))
    router_names = [name for name, _ in model.named_parameters() if ".router." in name]
    assert router_names
    assert all(name not in optimizer.partition.muon_names for name in router_names)
    assert all(name in optimizer.partition.adam_no_decay_names for name in router_names)
