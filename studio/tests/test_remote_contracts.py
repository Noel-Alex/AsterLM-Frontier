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
