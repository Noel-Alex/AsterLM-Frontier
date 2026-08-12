from __future__ import annotations

import torch
import yaml

from asterlm import AsterConfig, AsterLM, TrainConfig
from asterlm.optim import build_hybrid_optimizer
from asterlm.training.checkpoint import (
    checkpoint_storage_usage,
    enforce_checkpoint_storage_budget,
    load_checkpoint,
    load_data_state,
    pin_kda_backend_from_checkpoint,
    prune_rolling_checkpoints,
    resolve_checkpoint,
    save_checkpoint,
    verify_checkpoint,
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
        data_state={"cursor": 17, "residual": [1, 2, 3]},
    )
    saved_config = yaml.safe_load((checkpoint / "model_config.yaml").read_text(encoding="utf-8"))["model"]
    assert saved_config["kda_backend"] == "torch"
    manifest = verify_checkpoint(checkpoint)
    assert manifest["status"] == "complete"
    assert manifest["resume_state"]["optimizer"] is True
    assert manifest["resume_state"]["data_pipeline"] is True
    assert manifest["data_state_file"] == "data_state.pt"
    data_state = torch.load(checkpoint / "data_state.pt", weights_only=False)
    assert data_state["cursor"] == 17
    assert load_data_state(checkpoint) == {"cursor": 17, "residual": [1, 2, 3]}
    assert {item["path"] for item in manifest["artifacts"]} >= {
        manifest["model_file"],
        "trainer_state.pt",
        "model_config.yaml",
        "train_config.yaml",
    }
    assert not list(tmp_path.glob(".*.partial-*"))
    assert (tmp_path / "latest.txt").read_text(encoding="utf-8").strip() == checkpoint.name
    assert resolve_checkpoint(tmp_path) == checkpoint

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


def test_checkpoint_hash_detects_corruption(tmp_path):
    import json

    import pytest

    model_config = tiny_config()
    train_config = TrainConfig(device="cpu", max_steps=2, warmup_steps=0)
    model = AsterLM(model_config)
    optimizer = build_hybrid_optimizer(model, train_config)
    checkpoint = save_checkpoint(
        tmp_path, 0, model, optimizer, model_config, train_config, tokens_seen=0, keep_last=1
    )
    manifest = json.loads((checkpoint / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    model_path = checkpoint / manifest["model_file"]
    with model_path.open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(RuntimeError, match="size mismatch|hash mismatch"):
        verify_checkpoint(checkpoint)


def test_deferred_pruning_preserves_checkpoints_until_explicit_commit(tmp_path):
    model_config = tiny_config()
    train_config = TrainConfig(device="cpu", max_steps=2, warmup_steps=0)
    model = AsterLM(model_config)
    optimizer = build_hybrid_optimizer(model, train_config)
    first = save_checkpoint(
        tmp_path,
        1,
        model,
        optimizer,
        model_config,
        train_config,
        tokens_seen=8,
        keep_last=1,
        prune=False,
    )
    second = save_checkpoint(
        tmp_path,
        2,
        model,
        optimizer,
        model_config,
        train_config,
        tokens_seen=16,
        keep_last=1,
        prune=False,
    )
    assert first.exists() and second.exists()
    removed = prune_rolling_checkpoints(tmp_path, keep_last=1)
    assert removed == [first]
    assert not first.exists() and second.exists()


def test_checkpoint_pyramid_keeps_dense_recent_and_sparse_history(tmp_path):
    checkpoints = []
    for step in range(1, 31):
        path = tmp_path / f"checkpoint-{step:08d}"
        path.mkdir()
        checkpoints.append(path)

    prune_rolling_checkpoints(tmp_path, keep_last=6, pyramid_levels=3)

    retained_steps = {
        int(path.name.removeprefix("checkpoint-"))
        for path in tmp_path.glob("checkpoint-*")
    }
    assert retained_steps == {6, 18, 24, 25, 26, 27, 28, 29, 30}


def test_checkpoint_budget_never_evicts_unverified_permanent_or_recent(tmp_path):
    import json

    for step in range(1, 5):
        checkpoint = tmp_path / f"checkpoint-{step:08d}"
        checkpoint.mkdir()
        (checkpoint / "checkpoint_manifest.json").write_text("{}", encoding="utf-8")
        (checkpoint / "payload.bin").write_bytes(b"x" * 1024)
        if step in {1, 2}:
            (checkpoint / "KEEP").write_text("milestone\n", encoding="utf-8")
    verification = tmp_path / "hub-verifications"
    verification.mkdir()
    (verification / "checkpoint-00000001.json").write_text(
        json.dumps({"status": "verified", "checkpoint": "checkpoint-00000001"}),
        encoding="utf-8",
    )
    (tmp_path / "latest.txt").write_text("checkpoint-00000004\n", encoding="utf-8")

    result = enforce_checkpoint_storage_budget(
        tmp_path,
        max_total_gib=1.5 / 1024 / 1024,
        keep_last=1,
    )

    assert not (tmp_path / "checkpoint-00000001").exists()
    assert (tmp_path / "checkpoint-00000002").exists()
    assert (tmp_path / "checkpoint-00000004").exists()
    assert result["within_budget"] is False
    usage = checkpoint_storage_usage(tmp_path)
    assert usage["checkpoint_count"] == 2
