from __future__ import annotations

import torch
import yaml

from asterlm import AsterConfig, AsterLM, TrainConfig
from asterlm.optim import build_hybrid_optimizer
from asterlm.training.checkpoint import (
    load_checkpoint,
    pin_kda_backend_from_checkpoint,
    save_checkpoint,
)


def tiny_config() -> AsterConfig:
    return AsterConfig(
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
    )


def test_checkpoint_round_trip(tmp_path):
    torch.manual_seed(3)
    model_config = tiny_config()
    train_config = TrainConfig(device="cpu", max_steps=2, warmup_steps=0)
    model = AsterLM(model_config)
    optimizer = build_hybrid_optimizer(model, train_config)
    ids = torch.randint(0, model_config.vocab_size, (1, 8))
    loss = model(ids, labels=ids, return_logits=False).loss
    assert loss is not None
    loss.backward()
    optimizer.step()
    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}

    checkpoint = save_checkpoint(
        tmp_path,
        step=1,
        model=model,
        optimizer=optimizer,
        model_config=model_config,
        train_config=train_config,
        tokens_seen=8,
        keep_last=1,
    )
    saved_config = yaml.safe_load((checkpoint / "model_config.yaml").read_text(encoding="utf-8"))["model"]
    assert saved_config["kda_backend"] == "torch"

    auto_config = tiny_config()
    auto_config.kda_backend = "auto"
    pin_kda_backend_from_checkpoint(auto_config, checkpoint)
    assert auto_config.kda_backend == "torch"

    restored = AsterLM(model_config)
    restored_optimizer = build_hybrid_optimizer(restored, train_config)
    step, tokens = load_checkpoint(restored, restored_optimizer, checkpoint, restore_rng=False)

    assert step == 1
    assert tokens == 8
    for name, value in restored.state_dict().items():
        assert torch.equal(value, expected[name]), name


def test_checkpoint_rejects_explicit_backend_mismatch(tmp_path):
    import pytest

    model_config = tiny_config()
    train_config = TrainConfig(device="cpu", max_steps=2, warmup_steps=0)
    model = AsterLM(model_config)
    optimizer = build_hybrid_optimizer(model, train_config)
    checkpoint = save_checkpoint(
        tmp_path, 0, model, optimizer, model_config, train_config, tokens_seen=0, keep_last=1
    )
    incompatible = tiny_config()
    incompatible.kda_backend = "fla"
    with pytest.raises(ValueError, match="Checkpoint requires"):
        pin_kda_backend_from_checkpoint(incompatible, checkpoint)
