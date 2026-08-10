from __future__ import annotations

import json

from asterlm import AsterConfig
from asterlm.experiments.quality import (
    audit_named_initialization,
    latest_complete_checkpoint,
    summarize_quality_run,
)


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
            "tokens_per_second": 100.0,
            "gpu_util_percent": 90.0,
            "cuda_peak_allocated_gb": 2.0,
            "wall_clock_total_seconds": 30.0,
        },
        {"tokens_seen": 4096, "eval_main_loss": 3.5, "eval_perplexity": 33.1},
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
    assert summary["median_training_tokens_per_second"] == 100.0
    assert latest_complete_checkpoint(run) == checkpoint
