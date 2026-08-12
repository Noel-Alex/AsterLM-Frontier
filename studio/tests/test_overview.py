from __future__ import annotations

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
