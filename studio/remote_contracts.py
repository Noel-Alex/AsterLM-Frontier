from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

CONTRACT_VERSION = 4
SAFE_ALIAS = re.compile(r"^[A-Za-z0-9_.-]+$")
SAFE_HUB_REPO = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
SAFE_HUB_REVISION = re.compile(r"^[A-Za-z0-9._/-]+$")
SAFE_ACCELERATOR = re.compile(r"^[A-Za-z0-9._:+!-]+$")
SAFE_LOCATION = re.compile(r"^[a-z0-9-]+$")
SAFE_PROVISIONING = {"STANDARD", "SPOT"}
REMOTE_PROVIDERS = {"modal", "gcp", "lightning", "huggingface_jobs", "skypilot"}
DECISION_DATA_FLAGS = {
    "cleaned",
    "exact_deduplicated",
    "near_deduplicated",
    "cross_source_deduplicated",
    "benchmark_decontaminated",
    "validation_split_disjoint",
    "pii_handled",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    if not files:
        raise ValueError(f"Contract directory contains no files: {path}")
    for candidate in files:
        if candidate.is_symlink():
            raise ValueError(f"Contract directories may not contain symlinks: {candidate}")
        relative = candidate.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(candidate.stat().st_size.to_bytes(8, "big"))
        digest.update(bytes.fromhex(_sha256(candidate)))
    return digest.hexdigest()


def _repo_file(root: Path, value: str) -> Path:
    path = (root / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Contract path must stay inside the repository: {value}") from exc
    if not path.is_file():
        raise ValueError(f"Contract input does not exist: {value}")
    return path


def _repo_checkpoint(root: Path, value: str) -> tuple[Path, str, str]:
    path = (root / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Contract path must stay inside the repository: {value}") from exc
    if path.is_dir():
        return path, "directory", _tree_sha256(path)
    if path.is_file():
        return path, "file", _sha256(path)
    raise ValueError(f"Contract resume checkpoint does not exist: {value}")


def _hub_checkpoint(payload: dict[str, Any], default_repo: str) -> dict[str, str] | None:
    path = str(payload.get("resume_hub_path") or "").strip().strip("/")
    if not path:
        return None
    if path.startswith(".") or ".." in Path(path).parts or not path.startswith("runs/"):
        raise ValueError("resume_hub_path must be a safe runs/... checkpoint folder")
    repo = str(payload.get("resume_hub_repo") or default_repo).strip()
    if not SAFE_HUB_REPO.fullmatch(repo):
        raise ValueError("resume_hub_repo must use namespace/repository form")
    revision = str(payload.get("resume_hub_revision") or "main").strip()
    if not SAFE_HUB_REVISION.fullmatch(revision) or ".." in revision:
        raise ValueError("resume_hub_revision contains unsafe characters")
    return {"repo_id": repo, "revision": revision, "path": path}


def _dataset_manifest(root: Path, data_config: Path) -> tuple[dict[str, str], bool] | None:
    config_payload = yaml.safe_load(data_config.read_text(encoding="utf-8")) or {}
    if not isinstance(config_payload, dict):
        return None
    section = config_payload.get("data", config_payload)
    manifest_value = section.get("manifest_path") if isinstance(section, dict) else None
    if not manifest_value:
        return None
    manifest = _repo_file(root, str(manifest_value))
    normalized = str(manifest.relative_to(root.resolve())).replace(os.sep, "/")
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Dataset manifest is unreadable: {manifest}") from exc
    pipeline = payload.get("pipeline") if isinstance(payload, dict) else None
    decision_grade = bool(
        payload.get("schema_version") == 1
        and payload.get("status") == "complete"
        and isinstance(pipeline, dict)
        and all(pipeline.get(flag) is True for flag in DECISION_DATA_FLAGS)
    )
    return {"path": normalized, "sha256": _sha256(manifest)}, decision_grade


def git_state(root: Path) -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    )
    return {"commit": commit, "dirty": dirty}


def build_contract(
    payload: dict[str, Any],
    *,
    root: Path,
    policy: dict[str, Any],
    providers: list[dict[str, Any]],
    repository: dict[str, Any] | None = None,
) -> dict[str, Any]:
    provider_id = str(payload.get("provider") or policy.get("preferred") or "")
    if provider_id not in REMOTE_PROVIDERS:
        raise ValueError(
            "Remote contract provider must be Modal, Google Cloud, Lightning, "
            "Hugging Face Jobs, or SkyPilot"
        )

    provider = next((row for row in providers if row.get("id") == provider_id), None)
    if provider is None:
        raise ValueError(f"Unknown provider: {provider_id}")

    profile = str(payload.get("profile_alias") or "").strip()
    if not profile or not SAFE_ALIAS.fullmatch(profile):
        raise ValueError("profile_alias is required and may contain only letters, digits, dot, underscore, and hyphen")
    known_profiles = [str(value) for value in provider.get("profiles") or []]
    if known_profiles and profile not in known_profiles:
        raise ValueError(f"Unknown {provider_id} profile alias: {profile}")

    timeout_minutes = int(payload.get("timeout_minutes", 180))
    if not 1 <= timeout_minutes <= 10_080:
        raise ValueError("timeout_minutes must be between 1 and 10080")

    estimated_spend = float(payload.get("estimated_spend_usd", 0.0))
    policy_ceiling = float(policy.get("max_spend_usd_per_job", 0.0))
    if not math.isfinite(estimated_spend) or estimated_spend < 0:
        raise ValueError("estimated_spend_usd must be finite and non-negative")
    if estimated_spend > policy_ceiling:
        raise ValueError(
            f"Estimated spend ${estimated_spend:.2f} exceeds the Studio ceiling ${policy_ceiling:.2f}"
        )
    if policy.get("require_cost_confirmation", True) and not payload.get("cost_confirmed", False):
        raise ValueError("Explicit cost confirmation is required by the current provider policy")

    hub_repo = str(payload.get("hub_repo") or "").strip()
    if not SAFE_HUB_REPO.fullmatch(hub_repo):
        raise ValueError("hub_repo is required in namespace/repository form")
    gpu = str(payload.get("gpu") or "").strip()
    if gpu and not SAFE_ACCELERATOR.fullmatch(gpu):
        raise ValueError("gpu contains unsafe characters")
    zone = str(payload.get("zone") or "").strip()
    if zone and not SAFE_LOCATION.fullmatch(zone):
        raise ValueError("zone contains unsafe characters")
    provisioning_model = str(payload.get("provisioning_model") or "").strip().upper()
    if provisioning_model and provisioning_model not in SAFE_PROVISIONING:
        raise ValueError("provisioning_model must be STANDARD or SPOT")

    inputs: dict[str, dict[str, str]] = {}
    command = ["python", "scripts/studio_train.py", "--mode", "pretrain"]
    input_paths: dict[str, Path] = {}
    for key, option in (("model", "--model"), ("train", "--train"), ("data", "--data")):
        relative = str(payload.get(key) or "")
        path = _repo_file(root, relative)
        normalized = str(path.relative_to(root.resolve())).replace(os.sep, "/")
        inputs[key] = {"path": normalized, "sha256": _sha256(path)}
        input_paths[key] = path
        command.extend([option, normalized])
    command.extend(["--hub-repo", hub_repo])
    command.append("--remote-durable")
    manifest_input = _dataset_manifest(root, input_paths["data"])
    dataset_manifest_decision_grade = False
    if manifest_input is not None:
        inputs["dataset_manifest"] = manifest_input[0]
        dataset_manifest_decision_grade = manifest_input[1]

    resume = str(payload.get("resume") or "").strip()
    hub_checkpoint = _hub_checkpoint(payload, hub_repo)
    if resume and hub_checkpoint:
        raise ValueError("Use either resume or resume_hub_path, not both")
    if resume:
        resume_path, resume_kind, resume_sha = _repo_checkpoint(root, resume)
        normalized_resume = str(resume_path.relative_to(root.resolve())).replace(os.sep, "/")
        inputs["resume"] = {
            "path": normalized_resume,
            "kind": resume_kind,
            "sha256": resume_sha,
        }
        command.extend(["--resume", normalized_resume])
    elif hub_checkpoint:
        command.extend(["--resume", "__ASTER_HUB_RESUME__"])

    repository = repository or git_state(root)
    blockers: list[str] = []
    if repository.get("dirty"):
        blockers.append("repository_dirty")
    if not provider.get("ready"):
        blockers.append("provider_not_ready")

    created = datetime.now(UTC).isoformat()
    identity = {
        "version": CONTRACT_VERSION,
        "provider": provider_id,
        "profile_alias": profile,
        "git_commit": repository.get("commit"),
        "inputs": inputs,
        "command": command,
        "timeout_minutes": timeout_minutes,
        "estimated_spend_usd": estimated_spend,
        "max_spend_usd": policy_ceiling,
        "hub_repo": hub_repo,
        "gpu": gpu or None,
        "zone": zone or None,
        "provisioning_model": provisioning_model or None,
        "dataset_manifest_decision_grade": dataset_manifest_decision_grade,
        "resume_hub": hub_checkpoint,
        "parent_run_id": payload.get("parent_run_id"),
        "wandb_project": payload.get("wandb_project"),
    }
    contract_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:20]
    return {
        "contract_id": contract_id,
        "created_utc": created,
        **identity,
        "repository_dirty": bool(repository.get("dirty")),
        "provider_ready_at_creation": bool(provider.get("ready")),
        "blockers": blockers,
        "status": "ready" if not blockers else "blocked",
        "dispatch_adapter": {
            "gcp": "gcloud_compute_v1",
            "modal": "modal_sandbox_v1",
        }.get(provider_id, "planned"),
    }


def persist_contract(folder: Path, contract: dict[str, Any]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{contract['contract_id']}.json"
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path


def list_contracts(folder: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(folder.glob("*.json"), reverse=True) if folder.is_dir() else []:
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            row["path"] = str(path)
            rows.append(row)
        except (OSError, json.JSONDecodeError):
            continue
    return rows
