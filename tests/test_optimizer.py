import torch

from asterlm import AsterConfig, AsterLM, TrainConfig
from asterlm.optim import build_hybrid_optimizer
from asterlm.optim.muon import zeropower_via_newton_schulz5


def test_newton_schulz_shape_and_finiteness():
    matrix = torch.randn(16, 8)
    result = zeropower_via_newton_schulz5(matrix)
    assert result.shape == matrix.shape
    assert torch.isfinite(result).all()


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
