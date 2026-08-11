from __future__ import annotations

import json

from studio import server


def test_provider_control_routes_one_recorded_gcp_instance(tmp_path, monkeypatch):
    remote_id = "aster-" + "a" * 20
    job = {
        "provider": "gcp",
        "profile_alias": "google-credit",
        "instance_name": remote_id,
        "selected": {"zone": "us-central1-a"},
    }
    (tmp_path / f"{remote_id}.json").write_text(json.dumps(job), encoding="utf-8")
    observed = {}

    monkeypatch.setattr(server, "REMOTE_JOB_ROOT", tmp_path)
    monkeypatch.setattr(server, "load_gcp_profile", lambda *args: "profile")

    def fake_control(**kwargs):
        observed.update(kwargs)
        return {"provider": "gcp", "status": "graceful_stop_requested"}

    monkeypatch.setattr(server, "control_gcp_instance", fake_control)
    result = server.provider_control(remote_id, mode="graceful")

    assert result["status"] == "graceful_stop_requested"
    assert observed == {
        "profile": "profile",
        "instance_name": remote_id,
        "zone": "us-central1-a",
        "mode": "graceful",
    }
    persisted = json.loads((tmp_path / f"{remote_id}.json").read_text(encoding="utf-8"))
    assert persisted["last_control"]["status"] == "graceful_stop_requested"


def test_provider_control_rejects_unrecorded_or_broad_target(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "REMOTE_JOB_ROOT", tmp_path)
    try:
        server.provider_control("../all-instances", mode="terminate")
    except ValueError as exc:
        assert "remote job id" in str(exc)
    else:
        raise AssertionError("unsafe remote target was accepted")
