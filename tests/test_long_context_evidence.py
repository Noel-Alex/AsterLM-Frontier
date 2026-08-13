from __future__ import annotations

from pathlib import Path

import pytest

from scripts.import_long_context_evidence import validate_summary


def _summary(checkpoint: Path) -> dict:
    return {
        "schema_version": 1,
        "status": "complete",
        "promotion_eligible_source": True,
        "source_provenance": {"dirty": False, "git_commit": "a" * 40},
        "checkpoint": str(checkpoint),
        "checkpoint_manifest": {
            "status": "complete",
            "reason": "complete",
            "tokens_seen": 92_000_000_000,
        },
        "lengths": [8192, 16384, 32768],
        "tasks": ["exact_key", "repeated_key", "two_hop"],
        "depths": [0.1, 0.5, 0.9],
        "repeats": 3,
        "exact_greedy_accuracy": 0.75,
        "mean_answer_nll": 0.5,
        "by_task": {
            task: {"exact_greedy_accuracy": 0.5}
            for task in ("exact_key", "repeated_key", "two_hop")
        },
    }


def test_stage1_retrieval_proof_is_checkpoint_and_quality_bound(tmp_path: Path) -> None:
    root = tmp_path / "stage1"
    checkpoint = root / "checkpoint-final"
    checkpoint.mkdir(parents=True)
    result = validate_summary(
        _summary(checkpoint),
        "long_context_retrieval",
        expected_checkpoint_root=root,
        min_exact_accuracy=0.5,
    )
    assert result["checkpoint_tokens"] == 92_000_000_000
    assert result["required_lengths"] == [8192, 16384, 32768]


def test_retrieval_proof_rejects_proxy_checkpoint_and_task_collapse(tmp_path: Path) -> None:
    expected = tmp_path / "stage1"
    summary = _summary(tmp_path / "proxy" / "checkpoint-final")
    with pytest.raises(ValueError, match="must evaluate a checkpoint"):
        validate_summary(
            summary,
            "long_context_retrieval",
            expected_checkpoint_root=expected,
            min_exact_accuracy=0.5,
        )
    summary = _summary(expected / "checkpoint-final")
    summary["by_task"]["two_hop"]["exact_greedy_accuracy"] = 0.0
    with pytest.raises(ValueError, match="two_hop"):
        validate_summary(
            summary,
            "long_context_retrieval",
            expected_checkpoint_root=expected,
            min_exact_accuracy=0.5,
        )
