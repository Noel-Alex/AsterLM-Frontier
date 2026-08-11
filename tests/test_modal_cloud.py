from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from asterlm.cloud import (
    build_modal_cache_stage_plan,
    build_modal_launch_plan,
    build_modal_qualification_plan,
    control_modal_sandbox,
    dispatch_modal_launch_plan,
    load_modal_profile,
)
from scripts.cloud import modal_submit
from scripts.cloud.modal_submit import _verify_dataset_cache

ROOT = Path(__file__).resolve().parents[1]


def _contract(tmp_path: Path, **overrides) -> tuple[dict, Path]:
    payload = {
        "contract_id": "a" * 20,
        "provider": "modal",
        "profile_alias": "noelalex404",
        "git_commit": "b" * 40,
        "timeout_minutes": 10,
        "estimated_spend_usd": 10.0,
        "gpu": "B300",
        "blockers": ["provider_not_ready"],
        "inputs": {
            "dataset_manifest": {"path": "data/clean/manifest.json", "sha256": "c" * 64}
        },
        "dataset_manifest_decision_grade": True,
    }
    payload.update(overrides)
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload, path


def test_default_modal_profile_is_safely_blocked_and_secret_free(tmp_path):
    contract, contract_path = _contract(tmp_path)
    profile = load_modal_profile(ROOT / "configs/providers/modal_boost.yaml", "noelalex404")
    plan = build_modal_launch_plan(
        contract, profile, root=ROOT, contract_path=contract_path
    )
    assert plan["status"] == "blocked"
    assert "modal_base_image_not_configured" in plan["blockers"]
    assert "modal_dispatch_disabled" in plan["blockers"]
    assert "modal_workspace_or_environment_budget_not_confirmed" in plan["blockers"]
    assert plan["volume_version"] == 2
    assert plan["volumes"]["/opt/aster/data"] == "aster-data-noelalex404"
    serialized = json.dumps(plan)
    assert "HF_TOKEN" not in serialized
    assert "WANDB_API_KEY" not in serialized
    assert [row["gpu"] for row in plan["attempts"]] == ["B300"]
    assert plan["billing_policy"]["container_count"] == 1
    assert plan["billing_policy"]["silent_gpu_fallback"] is False
    assert plan["cache_policy"]["required_dataset_manifest"] == {
        "repository_path": "data/clean/manifest.json",
        "volume_path": "clean/manifest.json",
        "sha256": "c" * 64,
    }
    assert "before creating a GPU Sandbox" in plan["cache_policy"]["allocation_gate"]
    assert plan["estimated_max_cost_usd"] == pytest.approx(1.7748)


def test_modal_profiles_do_not_share_persistent_volumes():
    profiles = [
        load_modal_profile(ROOT / "configs/providers/modal_boost.yaml", alias)
        for alias in ("noelalex404", "friend-1", "friend-2")
    ]
    for field in ("dataset_volume", "cache_volume", "checkpoint_volume"):
        assert len({getattr(profile, field) for profile in profiles}) == len(profiles)


def test_modal_rejects_unpinned_image_and_incompatible_candidate(tmp_path):
    config = yaml.safe_load((ROOT / "configs/providers/modal_boost.yaml").read_text(encoding="utf-8"))
    raw = config["profiles"]["noelalex404"]
    raw["base_image"] = "nvcr.io/nvidia/pytorch:latest"
    raw["cuda_minor"] = "13.0"
    config_path = tmp_path / "modal.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    contract, contract_path = _contract(tmp_path, blockers=[])
    profile = load_modal_profile(config_path, "noelalex404")
    plan = build_modal_launch_plan(
        contract, profile, root=ROOT, contract_path=contract_path
    )
    assert "modal_base_image_not_digest_pinned" in plan["blockers"]
    assert plan["attempts"][0]["blockers"] == ["requires_cuda_13.1_or_newer"]


def test_modal_dispatch_is_dry_run_by_default(tmp_path):
    contract, contract_path = _contract(tmp_path)
    profile = load_modal_profile(ROOT / "configs/providers/modal_boost.yaml", "noelalex404")
    plan = build_modal_launch_plan(
        contract, profile, root=ROOT, contract_path=contract_path
    )
    result = dispatch_modal_launch_plan(plan)
    assert result["status"] == "dry_run"


class _FakeVolume:
    def __init__(self, chunks=None, error: Exception | None = None):
        self.chunks = chunks or []
        self.error = error

    async def read_file(self, path):
        if self.error is not None:
            raise self.error
        for chunk in self.chunks:
            yield chunk


def test_modal_cache_gate_hashes_the_persistent_manifest_before_allocation():
    payload = b'{"schema_version":1,"status":"complete"}'
    requirement = {
        "volume_path": "clean/manifest.json",
        "sha256": __import__("hashlib").sha256(payload).hexdigest(),
    }
    result = asyncio.run(
        _verify_dataset_cache(_FakeVolume([payload[:7], payload[7:]]), requirement)
    )
    assert result["size_bytes"] == len(payload)
    assert result["sha256"] == requirement["sha256"]


def test_modal_cache_gate_fails_closed_on_absent_or_stale_data():
    requirement = {"volume_path": "clean/manifest.json", "sha256": "a" * 64}
    with pytest.raises(RuntimeError, match="absent or unreadable"):
        asyncio.run(
            _verify_dataset_cache(_FakeVolume(error=FileNotFoundError()), requirement)
        )
    with pytest.raises(RuntimeError, match="hash mismatch"):
        asyncio.run(_verify_dataset_cache(_FakeVolume([b"stale"]), requirement))


def test_modal_submit_never_reaches_image_or_sandbox_when_cache_gate_fails(
    tmp_path, monkeypatch
):
    import hashlib

    import modal

    payload = tmp_path / "contract.json"
    payload.write_text("{}", encoding="utf-8")
    plan = {
        "provider": "modal",
        "job_kind": "training",
        "blockers": [],
        "profile_alias": "student",
        "contract_path": str(payload),
        "contract_sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
        "modal_environment": "main",
        "volume_version": 2,
        "volumes": {"/opt/aster/data": "aster-data-student"},
        "cache_policy": {
            "required_dataset_manifest": {
                "volume_path": "clean/manifest.json",
                "sha256": "a" * 64,
            }
        },
    }
    reached = {"image": False, "sandbox": False}

    def forbidden_image(*args, **kwargs):
        reached["image"] = True
        raise AssertionError("image construction occurred before the cache gate")

    def forbidden_sandbox(*args, **kwargs):
        reached["sandbox"] = True
        raise AssertionError("Sandbox allocation occurred before the cache gate")

    monkeypatch.setattr(modal.Volume, "from_name", lambda *args, **kwargs: _FakeVolume([b"stale"]))
    monkeypatch.setattr(modal.Image, "from_registry", forbidden_image)
    monkeypatch.setattr(modal.Sandbox, "create", forbidden_sandbox)
    monkeypatch.setenv("MODAL_PROFILE", "student")
    monkeypatch.setattr(sys, "argv", ["modal_submit.py", "--plan", str(tmp_path / "plan.json")])
    (tmp_path / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        modal_submit.main()
    assert reached == {"image": False, "sandbox": False}


def test_modal_cache_stage_plan_contains_only_sealed_clean_artifacts(tmp_path):
    import hashlib

    clean = tmp_path / "data" / "clean"
    clean.mkdir(parents=True)
    artifact = clean / "train-000.jsonl"
    artifact.write_bytes(b'{"text":"sealed"}\n')
    config = clean / "pretrain_data.yaml"
    config.write_text("data:\n  sources: []\n", encoding="utf-8")
    raw = tmp_path / "data" / "corpus-frontier-16b" / "raw.jsonl"
    raw.parent.mkdir(parents=True)
    raw.write_text("must-not-upload", encoding="utf-8")
    manifest = clean / "clean_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "path_base_hint": str(tmp_path),
                "data_config_path": "data/clean/pretrain_data.yaml",
                "data_config_file_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                "pipeline": {
                    flag: True
                    for flag in (
                        "cleaned",
                        "exact_deduplicated",
                        "near_deduplicated",
                        "cross_source_deduplicated",
                        "benchmark_decontaminated",
                        "validation_split_disjoint",
                        "pii_handled",
                    )
                },
                "artifacts": [
                    {
                        "path": "data/clean/train-000.jsonl",
                        "size_bytes": artifact.stat().st_size,
                        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    plan = build_modal_cache_stage_plan(
        manifest,
        root=tmp_path,
        profile_alias="student",
        modal_environment="main",
        volume_name="aster-data-student",
        verify_local_hashes=True,
    )
    assert plan["local_hashes_verified"] is True
    assert plan["raw_corpus_included"] is False
    assert plan["files"][-1]["role"] == "commit_marker"
    assert {row["volume_path"] for row in plan["files"]} == {
        "clean/train-000.jsonl",
        "clean/pretrain_data.yaml",
        "clean/clean_manifest.json",
    }
    assert all("corpus-frontier-16b" not in row["local_path"] for row in plan["files"])


def test_modal_qualification_is_one_bounded_synthetic_data_sandbox():
    profile = load_modal_profile(ROOT / "configs/providers/modal_boost.yaml", "noelalex404")
    plan = build_modal_qualification_plan(
        ROOT / "configs/providers/modal_qualification.json",
        profile,
        root=ROOT,
        gpu="L40S",
        timeout_minutes=30,
        estimated_spend_usd=1.0,
    )
    assert plan["job_kind"] == "qualification"
    assert plan["billing_policy"]["container_count"] == 1
    assert plan["billing_policy"]["silent_gpu_fallback"] is False
    assert plan["billing_policy"]["normal_exit_stops_billing"] is True
    assert plan["requested_gpu"] == "L40S"
    assert plan["secrets"] == []
    assert "/opt/aster/data" not in plan["volumes"]
    assert "no corpus" in plan["cache_policy"]["dataset"]
    assert "modal_dispatch_disabled" in plan["blockers"]


def test_modal_requires_explicit_gpu_and_honest_timeout_cost(tmp_path):
    contract, contract_path = _contract(
        tmp_path,
        gpu=None,
        timeout_minutes=180,
        estimated_spend_usd=1.0,
        blockers=[],
    )
    profile = load_modal_profile(ROOT / "configs/providers/modal_boost.yaml", "noelalex404")
    plan = build_modal_launch_plan(
        contract, profile, root=ROOT, contract_path=contract_path
    )
    assert "modal_gpu_must_be_explicitly_selected_in_contract" in plan["blockers"]

    contract["gpu"] = "H100!"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    plan = build_modal_launch_plan(
        contract, profile, root=ROOT, contract_path=contract_path
    )
    assert [row["gpu"] for row in plan["attempts"]] == ["H100!"]
    assert "modal_declared_spend_below_timeout_cost_guard" in plan["blockers"]


def test_modal_control_is_profile_isolated_and_structured(monkeypatch):
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["environment"] = kwargs["env"]
        payload = {
            "status": "graceful_stop_requested",
            "sandbox_id": "sb-AbC123",
        }
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="ASTER_MODAL_CONTROL_RESULT=" + json.dumps(payload) + "\n",
            stderr="",
        )

    monkeypatch.setattr("asterlm.cloud.modal.subprocess.run", fake_run)
    result = control_modal_sandbox(
        profile_alias="noelalex404",
        modal_environment="main",
        sandbox_id="sb-AbC123",
        mode="graceful",
    )
    assert result["status"] == "graceful_stop_requested"
    assert observed["environment"]["MODAL_PROFILE"] == "noelalex404"
    assert observed["environment"]["MODAL_ENVIRONMENT"] == "main"
    assert observed["command"][-1] == "graceful"


def test_modal_control_rejects_unknown_targets():
    with pytest.raises(ValueError, match="Sandbox id"):
        control_modal_sandbox(
            profile_alias="noelalex404",
            modal_environment="main",
            sandbox_id="not-a-sandbox",
            mode="terminate",
        )
