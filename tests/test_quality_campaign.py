from __future__ import annotations

import json

import pytest

from asterlm import AsterConfig
from asterlm.experiments import quality
from asterlm.experiments.quality import (
    archive_incomplete_quality_run,
    audit_identical_model_initialization,
    audit_named_initialization,
    latest_complete_checkpoint,
    summarize_quality_run,
)


def load_runner_module():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parents[1] / "scripts" / "run_architecture_quality_campaign.py"
    spec = importlib.util.spec_from_file_location("run_architecture_quality_campaign", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tiny(**changes) -> AsterConfig:
    values = {
        "vocab_size": 64,
        "d_model": 32,
        "n_layers": 2,
        "n_heads": 2,
        "head_dim": 16,
        "ffn_hidden": 64,
        "max_seq_len": 32,
        "kda_ratio": 1,
        "kda_backend": "torch",
        "latent_rank": 8,
        "rope_dim": 8,
        "mtp_depth": 0,
        "gradient_checkpointing": False,
    }
    values.update(changes)
    return AsterConfig(**values)


def test_named_initialization_audit_accepts_shared_projection_parity():
    report = audit_named_initialization(
        [
            ("compact", tiny()),
            ("wide-state", tiny(kda_num_heads=1, kda_head_dim=32)),
        ],
        1337,
    )
    assert report["status"] == "ok"
    assert report["candidates"]["wide-state"]["shared_parameter_count"] > 0
    assert report["candidates"]["wide-state"]["mismatches"] == []


def test_named_initialization_audit_releases_each_temporary_model(monkeypatch):
    releases = 0

    def record_release() -> None:
        nonlocal releases
        releases += 1

    monkeypatch.setattr(quality, "_release_initialization_audit_memory", record_release)
    audit_named_initialization(
        [("reference", tiny()), ("candidate-a", tiny()), ("candidate-b", tiny())],
        2027,
    )
    assert releases == 3


def test_full_initialization_audit_matches_optimizer_arms():
    report = audit_identical_model_initialization(
        ["fp32-state", "int8-state"], tiny(), 1337
    )
    assert report["status"] == "ok"
    assert report["scope"] == "all_unique_named_parameters"
    control = report["variants"]["fp32-state"]
    candidate = report["variants"]["int8-state"]
    assert control["tensor_count"] > 0
    assert candidate["fingerprint_sha256"] == control["fingerprint_sha256"]
    assert candidate["mismatches_vs_reference"] == []


def test_quality_summary_and_latest_checkpoint(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "experiment.json").write_text(
        json.dumps({"status": "ok", "completed_tokens": 4096, "run_id": "r1"}),
        encoding="utf-8",
    )
    rows = [
        {
            "tokens_seen": 2048,
            "step": 1,
            "tokens_per_second": 100.0,
            "gpu_util_percent": 90.0,
            "cuda_peak_allocated_gb": 2.0,
            "wall_clock_total_seconds": 30.0,
            "loss": 4.0,
            "grad_norm_pre_clip": 2.0,
            "grad_was_clipped": 1,
            "optimizer_submit_seconds": 3.0,
            "window_seconds": 10.0,
            "param_global_rms": 0.5,
        },
        {
            "tokens_seen": 3072,
            "step": 2,
            "tokens_per_second": 110.0,
            "wall_clock_total_seconds": 40.0,
            "loss": 3.0,
            "grad_norm_pre_clip": 1.0,
            "grad_was_clipped": 0,
            "optimizer_submit_seconds": 2.0,
            "window_seconds": 10.0,
            "param_global_rms": 0.55,
        },
        {"tokens_seen": 4096, "step": 2, "eval_main_loss": 3.5, "eval_perplexity": 33.1},
    ]
    (run / "metrics.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    checkpoint = run / "checkpoint-00000001"
    checkpoint.mkdir()
    (checkpoint / "checkpoint_manifest.json").write_text("{}", encoding="utf-8")
    (run / "latest.txt").write_text(str(checkpoint), encoding="utf-8")

    summary = summarize_quality_run(run)
    assert summary["eval_main_loss"] == 3.5
    assert summary["median_training_tokens_per_second"] == 105.0
    assert summary["learning_curve"][0]["wall_clock_total_seconds"] == 40.0
    assert summary["gradient_norm_p95"] == 1.95
    assert summary["gradient_clip_fraction"] == 0.5
    assert summary["parameter_global_rms_relative_drift"] == pytest.approx(0.1)
    assert summary["optimizer_wall_fraction_mean"] == 0.25
    assert latest_complete_checkpoint(run) == checkpoint


def test_interrupted_metrics_only_attempt_is_archived_atomically(tmp_path):
    run = tmp_path / "seed-7" / "candidate" / "bf16"
    run.mkdir(parents=True)
    (run / "experiment.json").write_text(
        json.dumps({"status": "running", "completed_tokens": 2048}),
        encoding="utf-8",
    )
    (run / "metrics.jsonl").write_text('{"tokens_seen": 2048}\n', encoding="utf-8")

    archive = archive_incomplete_quality_run(run, tmp_path / "interrupted")

    assert not run.exists()
    assert archive.parent == tmp_path / "interrupted"
    assert json.loads((archive / "experiment.json").read_text(encoding="utf-8"))[
        "completed_tokens"
    ] == 2048
    assert (archive / "metrics.jsonl").is_file()


def test_explicit_execution_matrix_accepts_heterogeneous_matched_pairs():
    runner = load_runner_module()
    materialized = {
        "candidates": {
            "dense": {"execution_variants": ["bf16"]},
            "sparse": {"execution_variants": ["reference", "fp8"]},
        }
    }
    candidates, matrix = runner._explicit_execution_matrix(
        materialized,
        ("dense=bf16", "sparse=reference", "sparse=fp8"),
    )
    assert candidates == ("dense", "sparse")
    assert matrix == [("dense", "bf16"), ("sparse", "reference"), ("sparse", "fp8")]


def test_resume_contract_rejects_changed_token_budget():
    runner = load_runner_module()
    existing = {
        "source_provenance": {"git_commit": "a" * 40},
        "candidates": ["dense"],
        "execution_matrix": [
            {"candidate_id": "dense", "execution_variant": "bf16"}
        ],
        "seeds": [7],
        "tokens_per_candidate": 4096,
        "smoke": False,
    }
    with pytest.raises(ValueError, match="tokens_per_candidate"):
        runner._validate_resume_contract(
            existing,
            candidates=("dense",),
            execution_matrix=[("dense", "bf16")],
            seeds=(7,),
            tokens=8192,
            smoke=False,
        )
