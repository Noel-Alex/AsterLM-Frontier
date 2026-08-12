from __future__ import annotations

from pathlib import Path

from scripts.run_moe_utilization_matrix import (
    median_moe_metric,
    median_moe_ratio,
    repository_provenance,
)


def test_repository_provenance_records_commit_status_and_diff_hash() -> None:
    repository_root = Path(__file__).resolve().parents[1]

    provenance = repository_provenance(repository_root)

    assert provenance["available"] is True
    assert len(provenance["commit"]) == 40
    assert isinstance(provenance["dirty"], bool)
    assert isinstance(provenance["status_porcelain"], list)
    assert len(provenance["tracked_diff_sha256"]) == 64
    assert provenance["tracked_diff_bytes"] >= 0


def test_repository_provenance_handles_non_repository(tmp_path: Path) -> None:
    provenance = repository_provenance(tmp_path)

    assert provenance["available"] is False
    assert provenance["root"] == str(tmp_path.resolve())
    assert provenance["error"]


def test_moe_aggregate_helpers_ignore_dense_summaries() -> None:
    summaries = [
        {},
        {"moe_execution": {}},
        {
            "moe_execution": {
                "moe_backend_forward_calls": 20,
                "moe_backend_host_metadata_syncs": 20,
            }
        },
        {
            "moe_execution": {
                "moe_backend_forward_calls": 40,
                "moe_backend_host_metadata_syncs": 0,
            }
        },
    ]

    assert median_moe_metric(summaries, "moe_backend_forward_calls") == 30
    assert median_moe_ratio(
        summaries,
        "moe_backend_host_metadata_syncs",
        "moe_backend_forward_calls",
    ) == 0.5
    assert median_moe_metric(summaries, "missing") is None
