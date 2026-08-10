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

CONTRACT_VERSION = 1
SAFE_ALIAS = re.compile(r"^[A-Za-z0-9_.-]+$")
SAFE_HUB_REPO = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
REMOTE_PROVIDERS = {"modal", "lightning", "huggingface_jobs", "skypilot"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repo_file(root: Path, value: str) -> Path:
    path = (root / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Contract path must stay inside the repository: {value}") from exc
    if not path.is_file():
        raise ValueError(f"Contract input does not exist: {value}")
    return path


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
        raise ValueError("Remote contract provider must be Modal, Lightning, Hugging Face Jobs, or SkyPilot")

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

    inputs: dict[str, dict[str, str]] = {}
    command = ["python", "scripts/studio_train.py", "--mode", "pretrain"]
    for key, option in (("model", "--model"), ("train", "--train"), ("data", "--data")):
        relative = str(payload.get(key) or "")
        path = _repo_file(root, relative)
        normalized = str(path.relative_to(root.resolve())).replace(os.sep, "/")
        inputs[key] = {"path": normalized, "sha256": _sha256(path)}
        command.extend([option, normalized])
    command.extend(["--hub-repo", hub_repo])

    resume = str(payload.get("resume") or "").strip()
    if resume:
        resume_path = _repo_file(root, resume)
        normalized_resume = str(resume_path.relative_to(root.resolve())).replace(os.sep, "/")
        inputs["resume"] = {"path": normalized_resume, "sha256": _sha256(resume_path)}
        command.extend(["--resume", normalized_resume])

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
        "dispatch_adapter": "planned",
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
