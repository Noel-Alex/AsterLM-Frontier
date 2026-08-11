from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SAFE_VALUE = re.compile(r"^[A-Za-z0-9._/@:-]+$")
SAFE_GPU = re.compile(r"^[A-Za-z0-9._:+!-]+$")
PINNED_IMAGE = re.compile(r"^[A-Za-z0-9._/-]+(?::[A-Za-z0-9._-]+)?@sha256:[0-9a-f]{64}$")
GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
PLACEHOLDER = re.compile(r"^SET_AFTER_")


@dataclass(frozen=True, slots=True)
class ModalProfile:
    alias: str
    modal_environment: str
    app_name: str
    base_image: str
    cuda_minor: str
    repository_url: str
    dataset_volume: str
    cache_volume: str
    checkpoint_volume: str
    volume_version: int
    hf_secret: str
    wandb_secret: str
    timeout_minutes: int
    dispatch_enabled: bool
    max_spend_usd_per_job: float
    spend_budget_confirmed: bool
    cost_contingency: float
    gpu_candidates: tuple[dict[str, Any], ...]


def _safe(value: str, field: str, pattern: re.Pattern[str] = SAFE_VALUE) -> str:
    if not value or not pattern.fullmatch(value):
        raise ValueError(f"Unsafe or empty Modal {field}")
    return value


def _cuda_number(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d+)\.(\d+)", value)
    if not match:
        raise ValueError("Modal cuda_minor must be major.minor")
    return int(match.group(1)), int(match.group(2))


def load_modal_profile(path: str | Path, alias: str) -> ModalProfile:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if payload.get("schema_version") != 1:
        raise ValueError("Modal provider config requires schema_version: 1")
    raw = (payload.get("profiles") or {}).get(alias)
    if not isinstance(raw, dict):
        raise KeyError(f"Unknown Modal profile: {alias}")
    candidates = tuple(dict(value) for value in raw.get("gpu_candidates", []))
    if not candidates:
        raise ValueError("Modal profile requires gpu_candidates")
    for candidate in candidates:
        _safe(str(candidate.get("gpu", "")), "GPU candidate", SAFE_GPU)
        if candidate.get("min_cuda") is not None:
            _cuda_number(str(candidate["min_cuda"]))
        rate = float(candidate.get("usd_per_second", 0.0))
        if rate <= 0:
            raise ValueError("Every Modal GPU candidate requires a positive usd_per_second")
    cuda_minor = str(raw.get("cuda_minor", ""))
    _cuda_number(cuda_minor)
    return ModalProfile(
        alias=_safe(alias, "profile alias"),
        modal_environment=_safe(str(raw.get("modal_environment", "main")), "environment"),
        app_name=_safe(str(raw["app_name"]), "app name"),
        base_image=str(raw["base_image"]),
        cuda_minor=cuda_minor,
        repository_url=_safe(str(raw["repository_url"]), "repository URL"),
        dataset_volume=_safe(str(raw["dataset_volume"]), "dataset volume"),
        cache_volume=_safe(str(raw["cache_volume"]), "cache volume"),
        checkpoint_volume=_safe(str(raw["checkpoint_volume"]), "checkpoint volume"),
        volume_version=int(raw.get("volume_version", 2)),
        hf_secret=_safe(str(raw["hf_secret"]), "HF secret"),
        wandb_secret=_safe(str(raw["wandb_secret"]), "W&B secret"),
        timeout_minutes=int(raw.get("timeout_minutes", 180)),
        dispatch_enabled=bool(raw.get("dispatch_enabled", False)),
        max_spend_usd_per_job=float(raw.get("max_spend_usd_per_job", 0.0)),
        spend_budget_confirmed=bool(raw.get("spend_budget_confirmed", False)),
        cost_contingency=float(raw.get("cost_contingency", 1.2)),
        gpu_candidates=candidates,
    )


def _global_blockers(profile: ModalProfile, contract: dict[str, Any]) -> list[str]:
    blockers = list(contract.get("blockers") or [])
    if PLACEHOLDER.match(profile.base_image):
        blockers.append("modal_base_image_not_configured")
    elif not PINNED_IMAGE.fullmatch(profile.base_image):
        blockers.append("modal_base_image_not_digest_pinned")
    if profile.volume_version != 2:
        blockers.append("modal_volume_v2_required_for_live_checkpoint_durability")
    if float(contract.get("estimated_spend_usd", 0.0)) > profile.max_spend_usd_per_job:
        blockers.append("modal_profile_spend_ceiling_exceeded")
    if not profile.dispatch_enabled:
        blockers.append("modal_dispatch_disabled")
    if not profile.spend_budget_confirmed:
        blockers.append("modal_workspace_or_environment_budget_not_confirmed")
    if not GIT_COMMIT.fullmatch(str(contract.get("git_commit", ""))):
        blockers.append("modal_contract_requires_exact_git_commit")
    if "dataset_manifest" not in (contract.get("inputs") or {}):
        blockers.append("modal_dataset_manifest_required_before_cache_or_training")
    elif not contract.get("dataset_manifest_decision_grade"):
        blockers.append("modal_decision_grade_dataset_manifest_required")
    return list(dict.fromkeys(blockers))


def build_modal_launch_plan(
    contract: dict[str, Any],
    profile: ModalProfile,
    *,
    root: str | Path,
    contract_path: str | Path | None = None,
) -> dict[str, Any]:
    if contract.get("provider") != "modal":
        raise ValueError("Modal launch plans require a modal remote contract")
    if contract.get("profile_alias") != profile.alias:
        raise ValueError("Modal profile alias does not match the remote contract")
    root = Path(root).resolve()
    submitter = root / "scripts" / "cloud" / "modal_submit.py"
    entrypoint = root / "scripts" / "cloud" / "modal_entrypoint.py"
    for path in (submitter, entrypoint, root / "scripts" / "cloud" / "run_contract.py"):
        if not path.is_file():
            raise FileNotFoundError(path)
    if contract_path is None:
        contract_path = (
            root / "data" / "aster-studio" / "remote-contracts" / f"{contract['contract_id']}.json"
        )
    contract_path = Path(contract_path).resolve()
    if not contract_path.is_file():
        raise FileNotFoundError(contract_path)
    contract_digest = hashlib.sha256(contract_path.read_bytes()).hexdigest()

    requested_gpu = str(contract.get("gpu") or "")
    attempts: list[dict[str, Any]] = []
    image_cuda = _cuda_number(profile.cuda_minor)
    for candidate in profile.gpu_candidates:
        if requested_gpu and str(candidate["gpu"]) != requested_gpu:
            continue
        blockers: list[str] = []
        minimum = candidate.get("min_cuda")
        if minimum is not None and image_cuda < _cuda_number(str(minimum)):
            blockers.append(f"requires_cuda_{minimum}_or_newer")
        attempts.append(
            {
                "gpu": str(candidate["gpu"]),
                "role": candidate.get("role"),
                "min_cuda": minimum,
                "usd_per_second": float(candidate["usd_per_second"]),
                "blockers": blockers,
            }
        )
    global_blockers = _global_blockers(profile, contract)
    if not requested_gpu:
        global_blockers.append("modal_gpu_must_be_explicitly_selected_in_contract")
    elif not attempts:
        global_blockers.append("modal_requested_gpu_is_not_in_profile_candidates")
    usable_attempts = [attempt for attempt in attempts if not attempt["blockers"]]
    if requested_gpu and not usable_attempts:
        global_blockers.append("modal_no_cuda_compatible_gpu_candidate")
    timeout_seconds = min(int(contract["timeout_minutes"]), profile.timeout_minutes) * 60
    estimated_max_cost = (
        float(usable_attempts[0]["usd_per_second"])
        * timeout_seconds
        * profile.cost_contingency
        if usable_attempts
        else None
    )
    if (
        estimated_max_cost is not None
        and float(contract.get("estimated_spend_usd", 0.0)) + 1e-9 < estimated_max_cost
    ):
        global_blockers.append("modal_declared_spend_below_timeout_cost_guard")
    blockers = list(dict.fromkeys(global_blockers))
    return {
        "schema_version": 1,
        "provider": "modal",
        "contract_id": contract["contract_id"],
        "contract_path": str(contract_path),
        "contract_sha256": contract_digest,
        "profile_alias": profile.alias,
        "modal_environment": profile.modal_environment,
        "app_name": profile.app_name,
        "timeout_seconds": timeout_seconds,
        "requested_gpu": requested_gpu or None,
        "estimated_max_cost_usd": estimated_max_cost,
        "image": {
            "base": profile.base_image,
            "cuda_minor": profile.cuda_minor,
            "repository_url": profile.repository_url,
            "git_commit": contract["git_commit"],
            "install_extras": ["cuda", "liger", "tracking", "frontier"],
        },
        "volumes": {
            "/opt/aster/data": profile.dataset_volume,
            "/var/cache/aster": profile.cache_volume,
            "/opt/aster/runs": profile.checkpoint_volume,
        },
        "volume_version": profile.volume_version,
        "secrets": [profile.hf_secret, profile.wandb_secret],
        "attempts": attempts,
        "blockers": blockers,
        "status": "dispatchable" if not blockers else "blocked",
        "secret_policy": "Modal secret names only; values never enter contracts, plans, or Studio",
        "billing_policy": {
            "container_count": 1,
            "silent_gpu_fallback": False,
            "pricing_researched_at": "2026-08-11",
            "contingency_multiplier": profile.cost_contingency,
            "hard_timeout_seconds": timeout_seconds,
            "contract_estimated_spend_usd": float(contract.get("estimated_spend_usd", 0.0)),
            "profile_ceiling_usd": profile.max_spend_usd_per_job,
        },
        "cache_policy": {
            "dataset": "profile-scoped Volume mounted at /opt/aster/data",
            "cache": "profile-scoped Volume mounted at /var/cache/aster",
            "checkpoint": "profile-scoped Volume v2 mounted at /opt/aster/runs",
            "cross_provider_resume": "verified private Hugging Face checkpoint folder",
        },
    }


def dispatch_modal_launch_plan(plan: dict[str, Any], *, execute: bool = False) -> dict[str, Any]:
    if not execute:
        return {"status": "dry_run", "plan": plan}
    if plan.get("blockers"):
        raise RuntimeError(f"Modal launch is blocked: {plan['blockers']}")
    root = Path(__file__).resolve().parents[3]
    submitter = root / "scripts" / "cloud" / "modal_submit.py"
    with tempfile.TemporaryDirectory(prefix="aster-modal-plan-") as folder:
        plan_path = Path(folder) / "plan.json"
        plan_path.write_text(json.dumps(plan, sort_keys=True) + "\n", encoding="utf-8")
        environment = dict(os.environ)
        environment["MODAL_PROFILE"] = str(plan["profile_alias"])
        environment["MODAL_ENVIRONMENT"] = str(plan["modal_environment"])
        completed = subprocess.run(
            [sys.executable, str(submitter), "--plan", str(plan_path)],
            cwd=root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
    if completed.returncode:
        raise RuntimeError(f"Modal dispatch failed: {completed.stderr.strip()[-4000:]}")
    marker = "ASTER_MODAL_RESULT="
    result_line = next(
        (line[len(marker) :] for line in reversed(completed.stdout.splitlines()) if line.startswith(marker)),
        None,
    )
    if result_line is None:
        raise RuntimeError("Modal dispatcher returned no structured result")
    return json.loads(result_line)


def control_modal_sandbox(
    *,
    profile_alias: str,
    modal_environment: str,
    sandbox_id: str,
    mode: str,
) -> dict[str, Any]:
    if mode not in {"status", "graceful", "terminate"}:
        raise ValueError("Modal control mode must be status, graceful, or terminate")
    if not re.fullmatch(r"sb-[A-Za-z0-9]+", sandbox_id):
        raise ValueError("Invalid Modal Sandbox id")
    root = Path(__file__).resolve().parents[3]
    controller = root / "scripts" / "cloud" / "modal_control.py"
    environment = dict(os.environ)
    environment["MODAL_PROFILE"] = _safe(profile_alias, "profile alias")
    environment["MODAL_ENVIRONMENT"] = _safe(modal_environment, "environment")
    completed = subprocess.run(
        [
            sys.executable,
            str(controller),
            "--sandbox-id",
            sandbox_id,
            "--mode",
            mode,
        ],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(f"Modal control failed: {completed.stderr.strip()[-4000:]}")
    marker = "ASTER_MODAL_CONTROL_RESULT="
    result_line = next(
        (line[len(marker) :] for line in reversed(completed.stdout.splitlines()) if line.startswith(marker)),
        None,
    )
    if result_line is None:
        raise RuntimeError("Modal controller returned no structured result")
    return json.loads(result_line)
