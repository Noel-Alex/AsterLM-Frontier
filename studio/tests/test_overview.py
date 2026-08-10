from __future__ import annotations

from studio import server


def test_overview_includes_clean_corpus_status(monkeypatch) -> None:
    monkeypatch.setattr(server, "dataset_status", lambda: [{"tokens": 7}])
    monkeypatch.setattr(server, "clean_corpora_status", lambda: [{"id": "clean-main"}])
    monkeypatch.setattr(server, "system_info", lambda: {"platform": "test"})
    monkeypatch.setattr(server, "runs_status", lambda: [])
    monkeypatch.setattr(server, "settings", lambda: {"providers": {}})
    monkeypatch.setattr(server, "provider_status", lambda _providers: [])
    monkeypatch.setattr(server.JOBS, "list", lambda: [])

    payload = server.overview()

    assert payload["clean_corpora"] == [{"id": "clean-main"}]
    assert payload["raw_materialized_tokens"] == 7
