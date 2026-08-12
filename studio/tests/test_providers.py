from __future__ import annotations

from studio import providers
from studio.server import DEFAULT_SETTINGS, validate_settings


def test_modal_profiles_are_aliases_only(tmp_path):
    config = tmp_path / ".modal.toml"
    config.write_text(
        '[student]\ntoken_id = "id-secret"\ntoken_secret = "do-not-leak"\n'
        '[friend-workspace]\ntoken_id = "other"\n',
        encoding="utf-8",
    )

    rows = providers.provider_status({"modal_config_path": str(config)})
    modal = next(row for row in rows if row["id"] == "modal")
    serialized = repr(modal)
    assert modal["profiles"] == ["friend-workspace", "student"]
    assert "do-not-leak" not in serialized
    assert "id-secret" not in serialized


def test_provider_catalog_has_sources_and_readiness():
    rows = providers.provider_status()
    assert {"local", "modal", "gcp", "lightning", "huggingface_jobs", "skypilot"} <= {
        row["id"] for row in rows
    }
    for row in rows:
        assert isinstance(row["ready"], bool)
        assert row["researched_at"]


def test_gcp_fallback_profile_never_implies_authentication(monkeypatch):
    monkeypatch.setattr(providers, "_command", lambda _name: None)
    row = next(
        item
        for item in providers.provider_status({"gcp_profiles": ["google-credit"]})
        if item["id"] == "gcp"
    )
    assert row["profiles"] == ["google-credit"]
    assert row["authenticated"] is False
    assert row["ready"] is False


def test_command_probe_keeps_virtualenv_sibling_path(tmp_path, monkeypatch):
    executable = tmp_path / "bin" / "python"
    executable.parent.mkdir()
    executable.touch()
    sibling = executable.parent / ("modal.exe" if providers.os.name == "nt" else "modal")
    sibling.touch()
    monkeypatch.setattr(providers.shutil, "which", lambda _name: None)
    monkeypatch.setattr(providers.sys, "executable", str(executable))

    assert providers._command("modal") == str(sibling)


def test_provider_policy_rejects_unknown_or_negative_spend():
    import pytest

    invalid = {**DEFAULT_SETTINGS, "providers": {"preferred": "made-up"}}
    with pytest.raises(ValueError, match="Unknown preferred provider"):
        validate_settings(invalid)

    invalid = {
        **DEFAULT_SETTINGS,
        "providers": {"preferred": "modal", "max_spend_usd_per_job": -1},
    }
    with pytest.raises(ValueError, match="non-negative"):
        validate_settings(invalid)
