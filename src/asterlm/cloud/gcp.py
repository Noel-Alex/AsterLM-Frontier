from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SAFE_VALUE = re.compile(r"^[A-Za-z0-9._/@:-]+$")
PLACEHOLDER = re.compile(r"^SET_AFTER_")
PINNED_IMAGE = re.compile(r"^[A-Za-z0-9._/-]+(?::[A-Za-z0-9._-]+)?@sha256:[0-9a-f]{64}$")
GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
SUPPORTED_PROVISIONING = {"STANDARD", "SPOT"}


@dataclass(frozen=True, slots=True)
class GcpProfile:
    alias: str
    gcloud_configuration: str
    project_id: str
    region: str
    zones: tuple[str, ...]
    host_image_project: str
    host_image: str
    host_image_id: str
    artifact_image: str
    bucket: str
    service_account: str
    billing_mode: str
    gpu_quota_confirmed: bool
    dispatch_enabled: bool
    max_spend_usd_per_job: float
    timeout_minutes: int
    cost_contingency: float
    provisioning_models: tuple[str, ...]
    machine_candidates: tuple[dict[str, Any], ...]
    boot_disk_gb: int
    checkpoint_prefix: str
    dataset_prefix: str
    hf_secret: str
    wandb_secret: str


def _safe(value: str, field: str) -> str:
    if not value or not SAFE_VALUE.fullmatch(value):
        raise ValueError(f"Unsafe or empty GCP {field}")
    return value


def load_gcp_profile(path: str | Path, alias: str) -> GcpProfile:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if payload.get("schema_version") != 1:
        raise ValueError("GCP provider config requires schema_version: 1")
    raw = (payload.get("profiles") or {}).get(alias)
    if not isinstance(raw, dict):
        raise KeyError(f"Unknown GCP profile: {alias}")
    provisioning = tuple(str(value).upper() for value in raw.get("provisioning_models", []))
    if not provisioning or set(provisioning) - SUPPORTED_PROVISIONING:
        raise ValueError("GCP provisioning_models must contain STANDARD and/or SPOT")
    candidates = tuple(dict(value) for value in raw.get("machine_candidates", []))
    if not candidates:
        raise ValueError("GCP profile requires machine_candidates")
    machine_types: set[str] = set()
    for candidate in candidates:
        machine_type = _safe(str(candidate.get("machine_type", "")), "machine type")
        if machine_type in machine_types:
            raise ValueError(f"Duplicate GCP machine candidate: {machine_type}")
        machine_types.add(machine_type)
        cost_guard = float(candidate.get("cost_guard_usd_per_hour", 0.0))
        if not math.isfinite(cost_guard) or cost_guard <= 0:
            raise ValueError("Every GCP machine candidate requires a finite positive cost guard")
    timeout_minutes = int(raw.get("timeout_minutes", 180))
    cost_contingency = float(raw.get("cost_contingency", 1.25))
    max_spend = float(raw.get("max_spend_usd_per_job", 0.0))
    boot_disk_gb = int(raw.get("boot_disk_gb", 200))
    if timeout_minutes <= 0:
        raise ValueError("GCP profile timeout_minutes must be positive")
    if not math.isfinite(cost_contingency) or cost_contingency < 1.0:
        raise ValueError("GCP profile cost_contingency must be finite and at least 1")
    if not math.isfinite(max_spend) or max_spend < 0:
        raise ValueError("GCP profile max_spend_usd_per_job must be finite and non-negative")
    if boot_disk_gb < 50:
        raise ValueError("GCP profile boot_disk_gb must be at least 50")
    zones = tuple(_safe(str(value), "zone") for value in raw.get("zones", []))
    if not zones or len(zones) != len(set(zones)):
        raise ValueError("GCP profile zones must be non-empty and unique")
    return GcpProfile(
        alias=_safe(alias, "profile alias"),
        gcloud_configuration=_safe(str(raw["gcloud_configuration"]), "configuration"),
        project_id=_safe(str(raw["project_id"]), "project"),
        region=_safe(str(raw["region"]), "region"),
        zones=zones,
        host_image_project=_safe(str(raw["host_image_project"]), "host image project"),
        host_image=_safe(str(raw["host_image"]), "host image"),
        host_image_id=_safe(str(raw["host_image_id"]), "host image id"),
        artifact_image=_safe(str(raw["artifact_image"]), "artifact image"),
        bucket=_safe(str(raw["bucket"]), "bucket"),
        service_account=_safe(str(raw["service_account"]), "service account"),
        billing_mode=str(raw.get("billing_mode", "free_trial_unactivated")),
        gpu_quota_confirmed=bool(raw.get("gpu_quota_confirmed", False)),
        dispatch_enabled=bool(raw.get("dispatch_enabled", False)),
        max_spend_usd_per_job=max_spend,
        timeout_minutes=timeout_minutes,
        cost_contingency=cost_contingency,
        provisioning_models=provisioning,
        machine_candidates=candidates,
        boot_disk_gb=boot_disk_gb,
        checkpoint_prefix=_safe(str(raw.get("checkpoint_prefix", "checkpoints")), "checkpoint prefix"),
        dataset_prefix=_safe(str(raw.get("dataset_prefix", "datasets")), "dataset prefix"),
        hf_secret=_safe(str(raw["hf_secret"]), "HF secret"),
        wandb_secret=_safe(str(raw["wandb_secret"]), "W&B secret"),
    )


def _profile_blockers(profile: GcpProfile, contract: dict[str, Any]) -> list[str]:
    blockers = list(contract.get("blockers") or [])
    for field, value in (
        ("project", profile.project_id),
        ("artifact_image", profile.artifact_image),
        ("bucket", profile.bucket),
        ("service_account", profile.service_account),
    ):
        if PLACEHOLDER.match(value):
            blockers.append(f"gcp_{field}_not_configured")
    if profile.billing_mode != "welcome_credit_upgraded":
        blockers.append("gcp_paid_billing_activation_required_for_gpu")
    if not profile.gpu_quota_confirmed:
        blockers.append("gcp_gpu_quota_not_confirmed")
    if float(contract.get("estimated_spend_usd", 0.0)) > profile.max_spend_usd_per_job:
        blockers.append("gcp_profile_spend_ceiling_exceeded")
    if not profile.dispatch_enabled:
        blockers.append("gcp_dispatch_disabled")
    if any(
        PLACEHOLDER.match(value)
        for value in (profile.host_image_project, profile.host_image, profile.host_image_id)
    ):
        blockers.append("gcp_qualified_host_image_required")
    if not profile.host_image_id.isdigit():
        blockers.append("gcp_host_image_numeric_id_required")
    if not PINNED_IMAGE.fullmatch(profile.artifact_image):
        blockers.append("gcp_artifact_image_must_be_digest_pinned")
    if not GIT_COMMIT.fullmatch(str(contract.get("git_commit") or "")):
        blockers.append("gcp_contract_requires_exact_git_commit")
    if "dataset_manifest" not in (contract.get("inputs") or {}):
        blockers.append("gcp_dataset_manifest_required")
    elif not contract.get("dataset_manifest_decision_grade"):
        blockers.append("gcp_decision_grade_dataset_manifest_required")
    return list(dict.fromkeys(blockers))


def _dataset_object_path(repository_path: str) -> str:
    normalized = repository_path.replace("\\", "/").strip("/")
    parts = normalized.split("/")
    if len(parts) < 2 or parts[0] != "data" or any(part in {"", ".", ".."} for part in parts):
        raise ValueError("GCP cached dataset inputs must be repository paths under data/")
    return "/".join(parts[1:])


def build_gcp_launch_plan(
    contract: dict[str, Any],
    profile: GcpProfile,
    *,
    root: str | Path,
    contract_path: str | Path | None = None,
) -> dict[str, Any]:
    if contract.get("provider") != "gcp":
        raise ValueError("GCP launch plans require a gcp remote contract")
    if contract.get("profile_alias") != profile.alias:
        raise ValueError("GCP profile alias does not match the remote contract")
    root = Path(root).resolve()
    startup = root / "scripts" / "cloud" / "gcp_startup.sh"
    shutdown = root / "scripts" / "cloud" / "gcp_shutdown.sh"
    for path in (startup, shutdown):
        if not path.is_file():
            raise FileNotFoundError(path)
    if contract_path is None:
        contract_path = (
            root
            / "data"
            / "aster-studio"
            / "remote-contracts"
            / f"{contract['contract_id']}.json"
        )
    contract_path = Path(contract_path).resolve()
    if not contract_path.is_file():
        raise FileNotFoundError(contract_path)
    contract_uri = (
        f"gs://{profile.bucket}/{profile.checkpoint_prefix}/contracts/"
        f"{contract['contract_id']}.json"
    )
    instance_name = f"aster-{contract['contract_id']}"
    upload = [
        "gcloud",
        "storage",
        "cp",
        str(contract_path),
        contract_uri,
        "--configuration",
        profile.gcloud_configuration,
        "--project",
        profile.project_id,
    ]
    requested_machine = str(contract.get("gpu") or "")
    requested_zone = str(contract.get("zone") or "")
    requested_provisioning = str(contract.get("provisioning_model") or "").upper()
    timeout_minutes = min(int(contract.get("timeout_minutes", 0)), profile.timeout_minutes)
    selected_candidate = next(
        (row for row in profile.machine_candidates if row["machine_type"] == requested_machine),
        None,
    )
    attempts: list[dict[str, Any]] = []
    if (
        selected_candidate is not None
        and requested_zone in profile.zones
        and requested_provisioning in profile.provisioning_models
    ):
        candidate = selected_candidate
        provisioning = requested_provisioning
        zone = requested_zone
        command = [
            "gcloud",
            "compute",
            "instances",
            "create",
            instance_name,
            "--configuration",
            profile.gcloud_configuration,
            "--project",
            profile.project_id,
            "--zone",
            zone,
            "--machine-type",
            str(candidate["machine_type"]),
            "--image",
            profile.host_image,
            "--image-project",
            profile.host_image_project,
            "--provisioning-model",
            provisioning,
            "--maintenance-policy",
            "TERMINATE",
            "--max-run-duration",
            f"{timeout_minutes}m",
            "--instance-termination-action",
            "DELETE",
            "--no-restart-on-failure",
            "--boot-disk-auto-delete",
            "--service-account",
            profile.service_account,
            "--scopes",
            "https://www.googleapis.com/auth/cloud-platform",
            "--boot-disk-size",
            f"{profile.boot_disk_gb}GB",
            "--metadata",
            (
                f"aster-contract-uri={contract_uri},"
                f"aster-artifact-image={profile.artifact_image},"
                f"aster-bucket={profile.bucket},aster-hf-secret={profile.hf_secret},"
                f"aster-wandb-secret={profile.wandb_secret},"
                f"aster-contract-id={contract['contract_id']},"
                f"aster-instance-name={instance_name},aster-instance-zone={zone},"
                f"aster-project-id={profile.project_id}"
            ),
            "--labels",
            f"aster-contract={contract['contract_id']},aster-job-kind=training",
            f"--metadata-from-file=startup-script={startup},shutdown-script={shutdown}",
        ]
        attempts.append(
            {
                "zone": zone,
                "machine_type": candidate["machine_type"],
                "accelerator": candidate.get("accelerator"),
                "role": candidate.get("role"),
                "provisioning_model": provisioning,
                "command": command,
            }
        )
    blockers = _profile_blockers(profile, contract)
    if not requested_machine:
        blockers.append("gcp_machine_type_must_be_explicit")
    elif selected_candidate is None:
        blockers.append("gcp_requested_machine_not_in_profile")
    if not requested_zone:
        blockers.append("gcp_zone_must_be_explicit")
    elif requested_zone not in profile.zones:
        blockers.append("gcp_requested_zone_not_in_profile")
    if not requested_provisioning:
        blockers.append("gcp_provisioning_model_must_be_explicit")
    elif requested_provisioning not in profile.provisioning_models:
        blockers.append("gcp_requested_provisioning_not_enabled")
    if timeout_minutes <= 0:
        blockers.append("gcp_timeout_must_be_positive")
    estimated_max_cost = (
        float(selected_candidate["cost_guard_usd_per_hour"])
        * timeout_minutes
        / 60.0
        * profile.cost_contingency
        if selected_candidate is not None and timeout_minutes > 0
        else None
    )
    if (
        estimated_max_cost is not None
        and float(contract.get("estimated_spend_usd", 0.0)) + 1e-9 < estimated_max_cost
    ):
        blockers.append("gcp_declared_spend_below_timeout_cost_guard")
    manifest = (contract.get("inputs") or {}).get("dataset_manifest")
    manifest_cache = None
    if isinstance(manifest, dict):
        try:
            object_path = _dataset_object_path(str(manifest["path"]))
            manifest_sha256 = str(manifest["sha256"])
            if not SHA256.fullmatch(manifest_sha256):
                raise ValueError("invalid dataset manifest SHA-256")
            manifest_cache = {
                "repository_path": str(manifest["path"]),
                "sha256": manifest_sha256,
                "uri": f"gs://{profile.bucket}/{profile.dataset_prefix}/{object_path}",
            }
        except (KeyError, ValueError):
            blockers.append("gcp_dataset_manifest_must_be_under_data_root")
    blockers = list(dict.fromkeys(blockers))
    return {
        "schema_version": 1,
        "provider": "gcp",
        "job_kind": str(contract.get("job_kind") or "training"),
        "contract_id": contract["contract_id"],
        "profile_alias": profile.alias,
        "gcloud_configuration": profile.gcloud_configuration,
        "project_id": profile.project_id,
        "instance_name": instance_name,
        "contract_upload": upload,
        "contract_uri": contract_uri,
        "contract_path": str(contract_path),
        "contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
        "host_image": {
            "project": profile.host_image_project,
            "name": profile.host_image,
            "expected_id": profile.host_image_id,
        },
        "timeout_minutes": timeout_minutes,
        "estimated_max_cost_usd": estimated_max_cost,
        "dataset_manifest_cache": manifest_cache,
        "attempts": attempts,
        "blockers": blockers,
        "status": "dispatchable" if not blockers else "blocked",
        "secret_policy": "Secret Manager names only; values never enter the contract or plan",
        "host_policy": (
            "exact custom image name and numeric image ID verified before allocation; "
            "moving image families are prohibited"
        ),
        "cache_policy": {
            "durable": f"gs://{profile.bucket}/{profile.dataset_prefix}",
            "container_dataset": "/opt/aster/data mounted read-only from the durable bucket prefix",
            "compiler": f"gs://{profile.bucket}/cache mounted at /var/cache/aster",
            "allocation_gate": "download and SHA-256 verify the clean manifest before VM creation",
            "checkpoint": f"gs://{profile.bucket}/{profile.checkpoint_prefix}",
        },
        "billing_policy": {
            "instance_count": 1,
            "silent_machine_or_zone_fallback": False,
            "hard_max_run_duration_minutes": timeout_minutes,
            "termination_action": "DELETE",
            "self_delete_on_process_exit": True,
            "cost_contingency": profile.cost_contingency,
            "declared_spend_usd": float(contract.get("estimated_spend_usd", 0.0)),
            "profile_ceiling_usd": profile.max_spend_usd_per_job,
        },
    }


def dispatch_gcp_launch_plan(plan: dict[str, Any], *, execute: bool = False) -> dict[str, Any]:
    if not execute:
        return {"status": "dry_run", "plan": plan}
    if plan.get("blockers"):
        raise RuntimeError(f"GCP launch is blocked: {plan['blockers']}")
    contract_path = Path(plan["contract_path"])
    if hashlib.sha256(contract_path.read_bytes()).hexdigest() != plan["contract_sha256"]:
        raise RuntimeError("GCP contract changed after launch-plan creation")
    host_image = plan.get("host_image")
    if not isinstance(host_image, dict):
        raise TypeError("GCP launch has no immutable host-image requirement")
    described_image = subprocess.run(
        [
            "gcloud",
            "compute",
            "images",
            "describe",
            host_image["name"],
            "--project",
            host_image["project"],
            "--configuration",
            plan["gcloud_configuration"],
            "--format=value(id)",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if described_image.returncode:
        raise RuntimeError("GCP qualified host image is absent or unreadable")
    observed_image_id = described_image.stdout.strip()
    if observed_image_id != host_image["expected_id"]:
        raise RuntimeError(
            "GCP host-image identity changed after qualification: "
            f"expected={host_image['expected_id']} observed={observed_image_id}"
        )
    cache = plan.get("dataset_manifest_cache")
    if not isinstance(cache, dict):
        raise TypeError("GCP launch has no dataset manifest cache requirement")
    cached_manifest = subprocess.run(
        [
            "gcloud", "storage", "cat", cache["uri"],
            "--configuration", plan["gcloud_configuration"],
            "--project", plan["project_id"],
        ],
        check=False,
        capture_output=True,
    )
    if cached_manifest.returncode:
        raise RuntimeError("GCP cached clean manifest is absent or unreadable")
    if len(cached_manifest.stdout) > 64 * 1024 * 1024:
        raise RuntimeError("GCP cached clean manifest exceeds the safety limit")
    observed = hashlib.sha256(cached_manifest.stdout).hexdigest()
    if observed != cache["sha256"]:
        raise RuntimeError(
            f"GCP cached clean manifest hash mismatch: expected={cache['sha256']} observed={observed}"
        )
    upload = subprocess.run(plan["contract_upload"], check=False, capture_output=True, text=True)
    if upload.returncode:
        raise RuntimeError(f"GCP contract upload failed: {upload.stderr.strip()}")
    failures: list[dict[str, Any]] = []
    for attempt in plan["attempts"]:
        completed = subprocess.run(attempt["command"], check=False, capture_output=True, text=True)
        if completed.returncode == 0:
            selected_keys = ("zone", "machine_type", "accelerator", "provisioning_model")
            return {
                "status": "dispatched",
                "provider": "gcp",
                "profile_alias": plan["profile_alias"],
                "contract_id": plan["contract_id"],
                "instance_name": plan["instance_name"],
                "selected": {key: attempt.get(key) for key in selected_keys},
                "stdout": completed.stdout.strip(),
                "failures_before_success": failures,
            }
        failures.append(
            {
                "zone": attempt["zone"],
                "machine_type": attempt["machine_type"],
                "provisioning_model": attempt["provisioning_model"],
                "error": completed.stderr.strip()[-2000:],
            }
        )
    raise RuntimeError("All GCP capacity attempts failed:\n" + json.dumps(failures, indent=2))


def control_gcp_instance(
    *,
    profile: GcpProfile,
    instance_name: str,
    zone: str,
    mode: str,
) -> dict[str, Any]:
    """Inspect or stop one exact Aster VM; never searches or broadens the target."""

    if mode not in {"status", "graceful", "terminate"}:
        raise ValueError("GCP control mode must be status, graceful, or terminate")
    if not re.fullmatch(r"aster-[0-9a-f]{20}", instance_name):
        raise ValueError("Invalid Aster GCP instance name")
    if zone not in profile.zones:
        raise ValueError("GCP instance zone is not enabled by the owning profile")
    common = [
        "--zone",
        zone,
        "--project",
        profile.project_id,
        "--configuration",
        profile.gcloud_configuration,
    ]
    if mode == "status":
        command = [
            "gcloud",
            "compute",
            "instances",
            "describe",
            instance_name,
            *common,
            "--format=json(name,status,zone,lastStartTimestamp,lastStopTimestamp,labels)",
        ]
    elif mode == "graceful":
        command = [
            "gcloud",
            "compute",
            "instances",
            "add-metadata",
            instance_name,
            *common,
            "--metadata=aster-stop-request=graceful",
        ]
    else:
        command = [
            "gcloud",
            "compute",
            "instances",
            "delete",
            instance_name,
            *common,
            "--delete-disks=all",
            "--quiet",
        ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(f"GCP instance control failed: {completed.stderr.strip()[-4000:]}")
    if mode == "status":
        try:
            details = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("GCP status returned invalid JSON") from exc
        return {"provider": "gcp", "status": "observed", "instance": details}
    if mode == "graceful":
        return {
            "provider": "gcp",
            "status": "graceful_stop_requested",
            "instance_name": instance_name,
            "checkpoint_policy": "optimizer-boundary checkpoint plus verified durable upload",
        }
    return {
        "provider": "gcp",
        "status": "terminated",
        "instance_name": instance_name,
        "warning": "Only checkpoints completed before termination are recoverable",
    }
