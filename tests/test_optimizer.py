import torch

from asterlm import AsterConfig, AsterLM, TrainConfig
from asterlm.optim import build_hybrid_optimizer, build_optimizer
from asterlm.optim.muon import (
    Muon,
    dequantize_blockwise_int8,
    quantize_blockwise_int8_,
    zeropower_via_newton_schulz5,
    zeropower_via_newton_schulz5_batched,
)


def test_blockwise_int8_muon_state_round_trip_and_tail():
    torch.manual_seed(11)
    source = torch.randn(3, 11, dtype=torch.float32)
    quantized, scales = quantize_blockwise_int8_(source.clone(), block_size=16)
    restored = dequantize_blockwise_int8(quantized, scales, block_size=16)
    assert quantized.dtype == torch.int8
    assert scales.shape == (3,)
    assert restored.shape == source.shape
    error = (restored - source).abs()
    expanded_scale = torch.cat(
        (scales[0].expand(16), scales[1].expand(16), scales[2].expand(1))
    ).reshape_as(source)
    assert torch.all(error <= expanded_scale.mul(0.501).add(1e-7))


def test_int8_muon_keeps_only_quantized_persistent_momentum():
    torch.manual_seed(13)
    parameter = torch.nn.Parameter(torch.randn(9, 17))
    parameter.grad = torch.randn_like(parameter)
    optimizer = Muon(
        [parameter], megabatch=True, state_dtype="int8_blockwise", quant_block_size=32
    )
    optimizer.step()
    state = optimizer.state[parameter]
    assert "momentum_buffer" not in state
    assert state["momentum_q"].dtype == torch.int8
    assert state["momentum_q"].shape == parameter.shape
    assert state["momentum_scale"].shape == (5,)


def test_int8_muon_checkpoint_resume_preserves_next_update():
    torch.manual_seed(23)
    reference = torch.nn.Parameter(torch.randn(9, 17, dtype=torch.bfloat16))
    resumed = torch.nn.Parameter(reference.detach().clone())
    optimizer = Muon(
        [reference], megabatch=True, state_dtype="int8_blockwise", quant_block_size=32
    )
    first_gradient = torch.randn_like(reference)
    reference.grad = first_gradient
    optimizer.step()
    resumed.data.copy_(reference)

    resumed_optimizer = Muon(
        [resumed], megabatch=True, state_dtype="int8_blockwise", quant_block_size=32
    )
    resumed_optimizer.load_state_dict(optimizer.state_dict())
    second_gradient = torch.randn_like(reference)
    reference.grad = second_gradient.clone()
    resumed.grad = second_gradient.clone()
    optimizer.step()
    resumed_optimizer.step()

    torch.testing.assert_close(resumed, reference, rtol=0, atol=0)
    original_state = optimizer.state[reference]
    restored_state = resumed_optimizer.state[resumed]
    torch.testing.assert_close(
        restored_state["momentum_q"], original_state["momentum_q"], rtol=0, atol=0
    )
    torch.testing.assert_close(
        restored_state["momentum_scale"],
        original_state["momentum_scale"],
        rtol=0,
        atol=0,
    )


def test_newton_schulz_shape_and_finiteness():
    matrix = torch.randn(16, 8)
    result = zeropower_via_newton_schulz5(matrix)
    assert result.shape == matrix.shape
    assert torch.isfinite(result).all()


def test_batched_newton_schulz_matches_independent_blocks():
    matrices = torch.randn(4, 8, 16)
    expected = torch.stack(
        [zeropower_via_newton_schulz5(matrix) for matrix in matrices]
    )
    actual = zeropower_via_newton_schulz5_batched(matrices)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_megabatched_muon_matches_parameterwise_state_and_updates():
    torch.manual_seed(17)
    reference = [torch.nn.Parameter(torch.randn(8, 16)) for _ in range(4)]
    candidate = [torch.nn.Parameter(parameter.detach().clone()) for parameter in reference]
    gradients = [torch.randn_like(parameter) for parameter in reference]
    for parameter, gradient in zip(reference, gradients, strict=True):
        parameter.grad = gradient.clone()
    for parameter, gradient in zip(candidate, gradients, strict=True):
        parameter.grad = gradient.clone()
    common = {
        "lr": 0.01,
        "momentum": 0.95,
        "weight_decay": 0.1,
        "ns_steps": 5,
        "nesterov": True,
        "update_rms": 0.2,
    }
    legacy = Muon(reference, megabatch=False, **common)
    batched = Muon(candidate, megabatch=True, megabatch_max_gib=0.001, **common)
    batched.set_diagnostics_enabled(True)
    legacy.step()
    batched.step()

    for expected, actual in zip(reference, candidate, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(
            batched.state[actual]["momentum_buffer"],
            legacy.state[expected]["momentum_buffer"],
        )
    diagnostics = batched.diagnostics()
    assert diagnostics["muon_megabatch_bucket_count"] == 1
    assert diagnostics["muon_megabatch_max_matrices"] > 1


def test_megabatched_muon_preserves_per_head_partition_math():
    torch.manual_seed(19)
    reference = torch.nn.Parameter(torch.randn(16, 8))
    candidate = torch.nn.Parameter(reference.detach().clone())
    gradient = torch.randn_like(reference)
    reference.grad = gradient.clone()
    candidate.grad = gradient.clone()
    group = lambda parameter: [{"params": [parameter], "split_count": 4}]
    legacy = Muon(group(reference), megabatch=False)
    batched = Muon(group(candidate), megabatch=True)
    legacy.step()
    batched.step()
    torch.testing.assert_close(candidate, reference, rtol=1e-5, atol=1e-6)


def test_hybrid_optimizer_step():
    config = AsterConfig(
        vocab_size=64,
        d_model=32,
        n_layers=2,
        n_heads=2,
        head_dim=16,
        ffn_hidden=96,
        max_seq_len=16,
        kda_ratio=1,
        kda_backend="torch",
        latent_rank=8,
        rope_dim=8,
        attention_window=None,
        sink_tokens=0,
        mtp_depth=1,
        mtp_rank=16,
        gradient_checkpointing=False,
        norm_type="ssnorm",
        embedding_projection=True,
    )
    model = AsterLM(config)
    optimizer = build_hybrid_optimizer(model, TrainConfig(device="cpu", max_steps=2))
    ids = torch.randint(0, 64, (2, 8))
    loss = model(ids, labels=ids).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    assert optimizer.partition.muon_names
    assert optimizer.partition.adam_no_decay_names
    assert "token_embedding.weight" in optimizer.partition.adam_no_decay_names
    assert "token_embedding.weight" not in optimizer.partition.muon_names
    assert "embedding_in_proj.weight" in optimizer.partition.muon_names
    assert "embedding_out_proj.weight" in optimizer.partition.muon_names


def test_adamw_control_has_no_muon_partition_and_steps_on_cpu():
    config = AsterConfig(
        vocab_size=64,
        d_model=32,
        n_layers=2,
        n_heads=2,
        head_dim=16,
        ffn_hidden=96,
        max_seq_len=16,
        kda_ratio=0,
        latent_rank=8,
        rope_dim=8,
        attention_window=None,
        sink_tokens=0,
        mtp_depth=0,
        gradient_checkpointing=False,
    )
    model = AsterLM(config)
    optimizer = build_optimizer(
        model, TrainConfig(device="cpu", max_steps=2, optimizer="adamw")
    )
    ids = torch.randint(0, 64, (2, 8))
    loss = model(ids, labels=ids).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    assert optimizer.partition.muon_names == []
    assert optimizer.partition.adam_decay_names
    assert "token_embedding.weight" in optimizer.partition.adam_no_decay_names


def test_per_head_muon_partitions_attention_projections_and_steps():
    config = AsterConfig(
        vocab_size=64,
        d_model=32,
        n_layers=2,
        n_heads=2,
        head_dim=16,
        ffn_hidden=96,
        max_seq_len=16,
        kda_ratio=1,
        kda_backend="torch",
        kda_num_heads=2,
        kda_head_dim=16,
        latent_rank=8,
        rope_dim=8,
        attention_window=None,
        sink_tokens=0,
        mtp_depth=0,
        gradient_checkpointing=False,
    )
    model = AsterLM(config)
    train = TrainConfig(
        device="cpu", max_steps=2, optimizer="muon_adamw", muon_per_head=True
    )
    optimizer = build_hybrid_optimizer(model, train)
    assert optimizer.partition.per_head_muon_names
    assert any("q_proj.weight" in name for name in optimizer.partition.per_head_muon_names)
    split_groups = [
        group for group in optimizer.muon.param_groups if group["split_count"] > 1
    ]
    assert split_groups and all(group["split_count"] == 2 for group in split_groups)
    ids = torch.randint(0, 64, (2, 8))
    loss = model(ids, labels=ids).loss
    loss.backward()
    optimizer.set_diagnostics_enabled(True)
    optimizer.step()
    diagnostics = optimizer.diagnostics()
    assert diagnostics["muon_matrix_count"] > 0
    assert diagnostics["muon_update_global_rms"] > 0
    assert diagnostics["muon_momentum_global_rms"] > 0
    assert diagnostics["muon_relative_update_rms_mean"] > 0


def test_per_head_muon_partitions_mla_qkv_up_projections():
    config = AsterConfig(
        vocab_size=64,
        d_model=32,
        n_layers=2,
        n_heads=2,
        head_dim=16,
        ffn_hidden=96,
        max_seq_len=16,
        kda_ratio=0,
        latent_rank=8,
        rope_dim=8,
        attention_window=None,
        sink_tokens=0,
        mtp_depth=0,
        gradient_checkpointing=False,
    )
    model = AsterLM(config)
    optimizer = build_hybrid_optimizer(
        model,
        TrainConfig(
            device="cpu", max_steps=2, optimizer="muon_adamw", muon_per_head=True
        ),
    )
    names = optimizer.partition.per_head_muon_names
    for projection in ("q_up.weight", "k_up.weight", "v_up.weight"):
        assert any(name.endswith(projection) for name in names)
    assert all(
        group["split_count"] == config.n_heads
        for group in optimizer.muon.param_groups
        if group["split_count"] > 1
    )
