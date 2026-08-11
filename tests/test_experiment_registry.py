from __future__ import annotations

import json

import pytest

from asterlm.experiments import ExperimentRegistry
from asterlm.training.engine import Trainer


def test_experiment_registry_lifecycle(tmp_path):
    registry = ExperimentRegistry.create(
        tmp_path / "run",
        repo_root=tmp_path,
        model={"d_model": 32},
        train={"max_tokens": 100, "tokenizer_path": "missing.json"},
        data={"train_sources": ["fixture"]},
        environment={"gpu": None},
        architecture={"parameters": 123},
        command=["python", "train.py"],
        hypothesis="fixture hypothesis",
    )
    run_id = registry.record["run_id"]
    registry.mark_running()
    registry.update_progress(40, throughput_tokens_s=200.0)
    registry.set_wandb_identity(
        entity="researcher",
        project="asterlm-frontier",
        run_id="stable-run-id",
        url="https://wandb.ai/researcher/asterlm-frontier/runs/stable-run-id",
    )
    registry.finish("ok", tokens_seen=100)

    saved = json.loads(registry.path.read_text(encoding="utf-8"))
    assert saved["run_id"] == run_id
    assert saved["status"] == "ok"
    assert saved["completion_fraction"] == 1.0
    assert saved["metrics"]["throughput_tokens_s"] == 200.0
    assert saved["metrics"]["wandb_run_id"] == "stable-run-id"
    assert saved["metrics"]["wandb_project"] == "asterlm-frontier"
    assert saved["hypothesis"] == "fixture hypothesis"


def test_registry_refuses_to_mix_fresh_runs_and_can_resume(tmp_path):
    kwargs = {
        "repo_root": tmp_path,
        "model": {"d_model": 32},
        "train": {"max_tokens": 100, "tokenizer_path": "missing.json"},
        "data": {"train_sources": ["fixture"]},
        "environment": {"gpu": None},
        "architecture": {"parameters": 123},
        "stage": "sft",
    }
    first = ExperimentRegistry.create(tmp_path / "run", **kwargs)
    assert first.record["stage"] == "sft"

    with pytest.raises(FileExistsError, match="already contains"):
        ExperimentRegistry.create(tmp_path / "run", **kwargs)

    resumed = ExperimentRegistry.create(tmp_path / "run", resume_existing=True, **kwargs)
    assert resumed.record["run_id"] == first.record["run_id"]


def test_wandb_identity_follows_checkpoint_to_a_new_output_directory(tmp_path):
    run = tmp_path / "source-run"
    registry = ExperimentRegistry.create(
        run,
        repo_root=tmp_path,
        model={"d_model": 32},
        train={"max_tokens": 100, "tokenizer_path": "missing.json"},
        data={"train_sources": ["fixture"]},
        environment={"gpu": None},
        architecture={"parameters": 123},
    )
    registry.set_wandb_identity(
        entity="researcher",
        project="asterlm-frontier",
        run_id="portable-wandb-run",
    )
    checkpoint = run / "checkpoint-00000010"
    checkpoint.mkdir()

    assert Trainer._stored_wandb_run_id(
        tmp_path / "new-provider-output",
        checkpoint_source=str(checkpoint),
    ) == "portable-wandb-run"
