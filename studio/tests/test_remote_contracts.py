from __future__ import annotations

import json

import pytest

from studio.remote_contracts import build_contract, persist_contract


def _payload() -> dict:
    return {
        "provider": "modal",
        "profile_alias": "student",
        "model": "model.yaml",
        "train": "train.yaml",
        "data": "data.yaml",
        "hub_repo": "student/aster-checkpoints",
        "timeout_minutes": 180,
        "estimated_spend_usd": 12,
        "cost_confirmed": True,
    }


def _providers(ready: bool = True) -> list[dict]:
    return [{"id": "modal", "ready": ready, "profiles": ["student"]}]


def test_gcp_contract_selects_real_dispatch_adapter(tmp_path):
    for name in ("model.yaml", "train.yaml", "data.yaml"):
        (tmp_path / name).write_text(name, encoding="utf-8")
    payload = _payload()
    payload.update({"provider": "gcp", "profile_alias": "google-credit"})
    contract = build_contract(
        payload,
        root=tmp_path,
        policy={"max_spend_usd_per_job": 30, "require_cost_confirmation": True},
        providers=[{"id": "gcp", "ready": False, "profiles": ["google-credit"]}],
        repository={"commit": "c" * 40, "dirty": False},
    )
    assert contract["dispatch_adapter"] == "gcloud_compute_v1"
    assert contract["blockers"] == ["provider_not_ready"]


def test_modal_contract_selects_real_dispatch_adapter(tmp_path):
    for name in ("model.yaml", "train.yaml", "data.yaml"):
        (tmp_path / name).write_text(name, encoding="utf-8")
    contract = build_contract(
        _payload(),
        root=tmp_path,
        policy={"max_spend_usd_per_job": 30, "require_cost_confirmation": True},
        providers=_providers(),
        repository={"commit": "d" * 40, "dirty": False},
    )
    assert contract["dispatch_adapter"] == "modal_sandbox_v1"


def test_contract_accepts_directory_resume_and_hashes_tree(tmp_path):
    for name in ("model.yaml", "train.yaml", "data.yaml"):
        (tmp_path / name).write_text(name, encoding="utf-8")
    checkpoint = tmp_path / "runs" / "trial" / "checkpoints" / "step-10"
    checkpoint.mkdir(parents=True)
    (checkpoint / "checkpoint_manifest.json").write_text("{}", encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    payload = _payload()
    payload["resume"] = "runs/trial/checkpoints/step-10"
    contract = build_contract(
        payload,
        root=tmp_path,
        policy={"max_spend_usd_per_job": 30, "require_cost_confirmation": True},
        providers=_providers(),
        repository={"commit": "e" * 40, "dirty": False},
    )
    assert contract["inputs"]["resume"]["kind"] == "directory"
    assert len(contract["inputs"]["resume"]["sha256"]) == 64


def test_contract_records_cross_provider_hub_resume(tmp_path):
    for name in ("model.yaml", "train.yaml", "data.yaml"):
        (tmp_path / name).write_text(name, encoding="utf-8")
    payload = _payload()
    payload.update(
        {
            "resume_hub_repo": "student/aster-checkpoints",
            "resume_hub_revision": "main",
            "resume_hub_path": "runs/trial/checkpoints/tokens-1000000",
        }
    )
    contract = build_contract(
        payload,
        root=tmp_path,
        policy={"max_spend_usd_per_job": 30, "require_cost_confirmation": True},
        providers=_providers(),
        repository={"commit": "f" * 40, "dirty": False},
    )
    assert contract["resume_hub"]["path"].endswith("tokens-1000000")
    assert "__ASTER_HUB_RESUME__" in contract["command"]
    assert "resume" not in contract["inputs"]


def test_contract_hashes_inputs_and_never_contains_credentials(tmp_path):
    for name in ("model.yaml", "train.yaml", "data.yaml"):
        (tmp_path / name).write_text(name, encoding="utf-8")
    contract = build_contract(
        _payload(),
        root=tmp_path,
        policy={"max_spend_usd_per_job": 30, "require_cost_confirmation": True},
        providers=_providers(),
        repository={"commit": "a" * 40, "dirty": False},
    )
    assert contract["status"] == "ready"
    assert contract["git_commit"] == "a" * 40
    assert all(len(row["sha256"]) == 64 for row in contract["inputs"].values())
    assert "token" not in json.dumps(contract).lower()
    path = persist_contract(tmp_path / "contracts", contract)
    assert json.loads(path.read_text(encoding="utf-8"))["contract_id"] == contract["contract_id"]


def test_contract_enforces_spend_confirmation_and_profile(tmp_path):
    for name in ("model.yaml", "train.yaml", "data.yaml"):
        (tmp_path / name).write_text(name, encoding="utf-8")
    payload = _payload()
    payload["cost_confirmed"] = False
    with pytest.raises(ValueError, match="cost confirmation"):
        build_contract(
            payload,
            root=tmp_path,
            policy={"max_spend_usd_per_job": 30, "require_cost_confirmation": True},
            providers=_providers(),
            repository={"commit": "a" * 40, "dirty": False},
        )
    payload = _payload()
    payload["estimated_spend_usd"] = 31
    with pytest.raises(ValueError, match="exceeds"):
        build_contract(
            payload,
            root=tmp_path,
            policy={"max_spend_usd_per_job": 30, "require_cost_confirmation": True},
            providers=_providers(),
            repository={"commit": "a" * 40, "dirty": False},
        )


def test_contract_records_readiness_and_clean_git_blockers(tmp_path):
    for name in ("model.yaml", "train.yaml", "data.yaml"):
        (tmp_path / name).write_text(name, encoding="utf-8")
    contract = build_contract(
        _payload(),
        root=tmp_path,
        policy={"max_spend_usd_per_job": 30, "require_cost_confirmation": True},
        providers=_providers(ready=False),
        repository={"commit": "b" * 40, "dirty": True},
    )
    assert contract["status"] == "blocked"
    assert set(contract["blockers"]) == {"repository_dirty", "provider_not_ready"}
