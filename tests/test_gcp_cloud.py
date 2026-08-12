from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from asterlm.cloud import (
    build_gcp_launch_plan,
    control_gcp_instance,
    dispatch_gcp_launch_plan,
    load_gcp_profile,
)
from asterlm.cloud.gcp_cache import build_gcp_cache_stage_plan, execute_gcp_cache_stage

ROOT = Path(__file__).resolve().parents[1]


def _contract(tmp_path: Path) -> tuple[dict, Path]:
    path = tmp_path / "contract.json"
    payload = {
        "contract_id": "a" * 20,
        "provider": "gcp",
        "profile_alias": "google-credit",
        "estimated_spend_usd": 10.0,
        "timeout_minutes": 10,
        "git_commit": "b" * 40,
        "gpu": "g2-standard-4",
        "zone": "us-central1-a",
        "provisioning_model": "STANDARD",
        "inputs": {
            "dataset_manifest": {
                "path": "data/clean/clean_manifest.json",
                "sha256": "c" * 64,
            }
        },
        "dataset_manifest_decision_grade": True,
        "blockers": ["provider_not_ready"],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload, path


def test_default_gcp_profile_is_safely_blocked_until_credentials_and_billing(tmp_path):
    profile = load_gcp_profile(ROOT / "configs/providers/gcp_boost.yaml", "google-credit")
    contract, contract_path = _contract(tmp_path)
    plan = build_gcp_launch_plan(
        contract, profile, root=ROOT, contract_path=contract_path
    )
    assert plan["status"] == "blocked"
    assert "gcp_paid_billing_activation_required_for_gpu" in plan["blockers"]
    assert "gcp_gpu_quota_not_confirmed" in plan["blockers"]
    assert "gcp_dispatch_disabled" in plan["blockers"]
    assert "gcp_qualified_host_image_required" in plan["blockers"]
    assert "gcp_artifact_image_must_be_digest_pinned" in plan["blockers"]
    serialized = json.dumps(plan)
    assert "HF_TOKEN" not in serialized
    assert "WANDB_API_KEY" not in serialized
    assert [row["accelerator"] for row in plan["attempts"]] == ["l4-24gb"]
    assert plan["billing_policy"]["instance_count"] == 1
    assert plan["billing_policy"]["silent_machine_or_zone_fallback"] is False
    assert plan["billing_policy"]["termination_action"] == "DELETE"
    command = plan["attempts"][0]["command"]
    assert command[command.index("--max-run-duration") + 1] == "10m"
    assert command[command.index("--instance-termination-action") + 1] == "DELETE"
    assert command[command.index("--image") + 1] == profile.host_image
    assert "--image-family" not in command
    assert plan["dataset_manifest_cache"]["uri"].endswith(
        "/datasets/clean/clean_manifest.json"
    )


def test_gcp_dispatch_is_dry_run_by_default(tmp_path):
    profile = load_gcp_profile(ROOT / "configs/providers/gcp_boost.yaml", "google-credit")
    contract, contract_path = _contract(tmp_path)
    plan = build_gcp_launch_plan(
        contract, profile, root=ROOT, contract_path=contract_path
    )
    result = dispatch_gcp_launch_plan(plan)
    assert result["status"] == "dry_run"


def test_gcp_cache_gate_fails_before_contract_upload_or_vm_creation(tmp_path, monkeypatch):
    contract = tmp_path / "contract.json"
    contract.write_text("{}", encoding="utf-8")
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[1:4] == ["compute", "images", "describe"]:
            return subprocess.CompletedProcess(command, 0, stdout="12345\n", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout=b"stale", stderr=b"")

    monkeypatch.setattr("asterlm.cloud.gcp.subprocess.run", fake_run)
    plan = {
        "provider": "gcp",
        "blockers": [],
        "contract_path": str(contract),
        "contract_sha256": hashlib.sha256(contract.read_bytes()).hexdigest(),
        "host_image": {
            "name": "aster-host-v1",
            "project": "student-project",
            "expected_id": "12345",
        },
        "dataset_manifest_cache": {
            "uri": "gs://bucket/datasets/clean/manifest.json",
            "sha256": "a" * 64,
        },
        "gcloud_configuration": "student",
        "project_id": "student-project",
    }
    with pytest.raises(RuntimeError, match="hash mismatch"):
        dispatch_gcp_launch_plan(plan, execute=True)
    assert len(calls) == 2
    assert calls[0][1:4] == ["compute", "images", "describe"]
    assert calls[1][:3] == ["gcloud", "storage", "cat"]


def test_gcp_host_image_identity_fails_before_cache_or_vm(tmp_path, monkeypatch):
    contract = tmp_path / "contract.json"
    contract.write_text("{}", encoding="utf-8")
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="99999\n", stderr="")

    monkeypatch.setattr("asterlm.cloud.gcp.subprocess.run", fake_run)
    plan = {
        "provider": "gcp",
        "blockers": [],
        "contract_path": str(contract),
        "contract_sha256": hashlib.sha256(contract.read_bytes()).hexdigest(),
        "host_image": {
            "name": "aster-host-v1",
            "project": "student-project",
            "expected_id": "12345",
        },
        "gcloud_configuration": "student",
    }
    with pytest.raises(RuntimeError, match="identity changed"):
        dispatch_gcp_launch_plan(plan, execute=True)
    assert len(calls) == 1
    assert calls[0][1:4] == ["compute", "images", "describe"]


@pytest.mark.parametrize(
    ("mode", "verb", "expected_status"),
    [
        ("graceful", "add-metadata", "graceful_stop_requested"),
        ("terminate", "delete", "terminated"),
    ],
)
def test_gcp_control_targets_one_exact_instance(
    monkeypatch, mode, verb, expected_status
):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("asterlm.cloud.gcp.subprocess.run", fake_run)
    profile = load_gcp_profile(ROOT / "configs/providers/gcp_boost.yaml", "google-credit")
    result = control_gcp_instance(
        profile=profile,
        instance_name="aster-" + "a" * 20,
        zone="us-central1-a",
        mode=mode,
    )
    assert result["status"] == expected_status
    assert calls[0][1:5] == ["compute", "instances", verb, "aster-" + "a" * 20]
    assert "--zone" in calls[0]
    assert "--quiet" in calls[0] if mode == "terminate" else True


def test_gcp_cache_staging_is_sealed_incremental_and_manifest_last(tmp_path, monkeypatch):
    clean = tmp_path / "data" / "clean"
    clean.mkdir(parents=True)
    artifact = clean / "train.jsonl"
    artifact.write_bytes(b'{"text":"clean"}\n')
    config = clean / "data.yaml"
    config.write_text("data:\n  sources: []\n", encoding="utf-8")
    manifest = clean / "clean_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "path_base_hint": str(tmp_path),
                "data_config_path": "data/clean/data.yaml",
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
                        "path": "data/clean/train.jsonl",
                        "size_bytes": artifact.stat().st_size,
                        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    profile = load_gcp_profile(ROOT / "configs/providers/gcp_boost.yaml", "google-credit")
    plan = build_gcp_cache_stage_plan(
        manifest, profile, root=tmp_path, verify_local_hashes=True
    )
    assert plan["raw_corpus_included"] is False
    assert plan["files"][-1]["role"] == "commit_marker"
    assert plan["files"][-1]["uri"].endswith("/datasets/clean/clean_manifest.json")

    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[2:4] == ["objects", "describe"]:
            uri = command[4]
            row = next(item for item in plan["files"] if item["uri"] == uri)
            if row["role"] == "artifact":
                body = {
                    "size": str(row["size_bytes"]),
                    "custom_fields": {"aster-sha256": row["sha256"]},
                }
                return subprocess.CompletedProcess(command, 0, stdout=json.dumps(body), stderr="")
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="missing")
        return subprocess.CompletedProcess(command, 0, stdout="uploaded", stderr="")

    monkeypatch.setattr("asterlm.cloud.gcp_cache.subprocess.run", fake_run)
    result = execute_gcp_cache_stage(plan)
    assert result["reused_files"] == 1
    assert result["uploaded_files"] == 2
    uploads = [command for command in calls if command[2] == "cp"]
    assert uploads[-1][4].endswith("/datasets/clean/clean_manifest.json")
