from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SAFE_VALUE = re.compile(r"^[A-Za-z0-9._/@:-]+$")
PLACEHOLDER = re.compile(r"^SET_AFTER_")
SUPPORTED_PROVISIONING = {"STANDARD", "SPOT"}


@dataclass(frozen=True, slots=True)
class GcpProfile:
    alias: str
    gcloud_configuration: str
    project_id: str
    region: str
    zones: tuple[str, ...]
    artifact_image: str
    bucket: str
    service_account: str
    billing_mode: str
    gpu_quota_confirmed: bool
    dispatch_enabled: bool
    max_spend_usd_per_job: float
    provisioning_models: tuple[str, ...]
    machine_candidates: tuple[dict[str, str], ...]
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
    for candidate in candidates:
        _safe(str(candidate.get("machine_type", "")), "machine type")
    return GcpProfile(
        alias=_safe(alias, "profile alias"),
        gcloud_configuration=_safe(str(raw["gcloud_configuration"]), "configuration"),
        project_id=_safe(str(raw["project_id"]), "project"),
        region=_safe(str(raw["region"]), "region"),
        zones=tuple(_safe(str(value), "zone") for value in raw.get("zones", [])),
        artifact_image=_safe(str(raw["artifact_image"]), "artifact image"),
        bucket=_safe(str(raw["bucket"]), "bucket"),
        service_account=_safe(str(raw["service_account"]), "service account"),
        billing_mode=str(raw.get("billing_mode", "free_trial_unactivated")),
        gpu_quota_confirmed=bool(raw.get("gpu_quota_confirmed", False)),
        dispatch_enabled=bool(raw.get("dispatch_enabled", False)),
        max_spend_usd_per_job=float(raw.get("max_spend_usd_per_job", 0.0)),
        provisioning_models=provisioning,
        machine_candidates=candidates,
        boot_disk_gb=int(raw.get("boot_disk_gb", 200)),
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
    return list(dict.fromkeys(blockers))


def build_gcp_launch_plan(
    contract: dict[str, Any],
    profile: GcpProfile,
    *,
    root: str | Path,
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
    contract_path = (
        root
        / "data"
        / "aster-studio"
        / "remote-contracts"
        / f"{contract['contract_id']}.json"
    )
    contract_uri = (
        f"gs://{profile.bucket}/{profile.checkpoint_prefix}/contracts/"
        f"{contract['contract_id']}.json"
    )
    instance_name = f"aster-{contract['contract_id'][:16]}"
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
    attempts: list[dict[str, Any]] = []
    for provisioning in profile.provisioning_models:
        for candidate in profile.machine_candidates:
            for zone in profile.zones:
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
                    "--provisioning-model",
                    provisioning,
                    "--maintenance-policy",
                    "TERMINATE",
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
                        f"aster-wandb-secret={profile.wandb_secret}"
                    ),
                    f"--metadata-from-file=startup-script={startup},shutdown-script={shutdown}",
                ]
                if provisioning == "SPOT":
                    command.extend(["--instance-termination-action", "STOP"])
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
    return {
        "schema_version": 1,
        "provider": "gcp",
        "contract_id": contract["contract_id"],
        "profile_alias": profile.alias,
        "project_id": profile.project_id,
        "instance_name": instance_name,
        "contract_upload": upload,
        "contract_uri": contract_uri,
        "attempts": attempts,
        "blockers": blockers,
        "status": "dispatchable" if not blockers else "blocked",
        "secret_policy": "Secret Manager names only; values never enter the contract or plan",
        "cache_policy": {
            "durable": f"gs://{profile.bucket}/{profile.dataset_prefix}",
            "local": "/var/lib/aster-cache",
            "checkpoint": f"gs://{profile.bucket}/{profile.checkpoint_prefix}",
        },
    }


def dispatch_gcp_launch_plan(plan: dict[str, Any], *, execute: bool = False) -> dict[str, Any]:
    if not execute:
        return {"status": "dry_run", "plan": plan}
    if plan.get("blockers"):
        raise RuntimeError(f"GCP launch is blocked: {plan['blockers']}")
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
