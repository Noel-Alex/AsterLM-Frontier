from __future__ import annotations

from pathlib import Path

import pytest

from studio import server


def test_overview_includes_clean_corpus_status(monkeypatch) -> None:
    monkeypatch.setattr(server, "dataset_status", lambda: [{"tokens": 7}])
    monkeypatch.setattr(server, "clean_corpora_status", lambda: [{"id": "clean-main"}])
    monkeypatch.setattr(server, "system_info", lambda: {"platform": "test"})
    monkeypatch.setattr(server, "runs_status", list)
    monkeypatch.setattr(server, "settings", lambda: {"providers": {}})
    monkeypatch.setattr(server, "provider_status", lambda _providers: [])
    monkeypatch.setattr(server, "execution_backend_status", lambda: [{"backend": "aster_local"}])
    monkeypatch.setattr(server.JOBS, "list", list)

    payload = server.overview()

    assert payload["clean_corpora"] == [{"id": "clean-main"}]
    assert payload["raw_materialized_tokens"] == 7
    assert payload["execution_backends"] == [{"backend": "aster_local"}]


def test_retired_stack_edu_is_rejected_from_active_workflows(monkeypatch) -> None:
    monkeypatch.setattr(server, "dataset_status", lambda: [])
    with pytest.raises(ValueError, match="retired"):
        server.create_clean_plan(
            {"sources": [{"id": "stack_edu", "tokens": 1, "allow_empty": True}]}
        )
    with pytest.raises(ValueError, match="retired"):
        server.start_action(
            "download_source", {"source_id": "stack_edu", "target_tokens": 1_000_000}
        )


def test_stale_campaign_preview_cannot_override_current_public_checkpoint_repo() -> None:
    campaign = {
        "name": "current",
        "goal_tokens": 100,
        "checkpointing": {"public_hub_repository": "owner/public"},
    }
    state = {
        "campaign": "configs/pretraining/frontier_100b_k3.yaml",
        "name": "current",
        "goal_tokens": 100,
        "hub_repo": "owner/private",
        "status": "dry_run",
        "commands": [["python", "obsolete.py"]],
    }

    result = server.current_supervisor_state(
        campaign,
        state,
        state_path=Path("runs/current-campaign/campaign_state.json"),
    )

    assert result["status"] == "stale_historical_state"
    assert result["current"] is False
    assert result["stale_reasons"] == ["checkpoint_repository_changed"]
    assert "commands" not in result
