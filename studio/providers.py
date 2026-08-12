from __future__ import annotations

import os
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

RESEARCHED_AT = "2026-08-11"


PROVIDER_CATALOG: dict[str, dict[str, Any]] = {
    "local": {
        "label": "Local WSL / CUDA",
        "kind": "native",
        "automation": "ready",
        "credit": "Owned hardware; no metered compute charge",
        "source_url": None,
    },
    "modal": {
        "label": "Modal",
        "kind": "job",
        "automation": "profile-aware",
        "credit": "$30/month Starter compute credit; academic grants up to $10k",
        "source_url": "https://modal.com/pricing",
    },
    "gcp": {
        "label": "Google Cloud Compute Engine",
        "kind": "gpu-vm-job",
        "automation": "gcloud-contract",
        "credit": "$300 Welcome credit; GPU access requires activating paid billing and quota",
        "source_url": "https://cloud.google.com/free/docs/free-cloud-features",
    },
    "lightning": {
        "label": "Lightning AI",
        "kind": "studio-job",
        "automation": "cli-sdk",
        "credit": "15 free credits/month; availability and GPU prices vary",
        "source_url": "https://lightning.ai/pricing",
    },
    "huggingface_jobs": {
        "label": "Hugging Face Jobs",
        "kind": "job",
        "automation": "cli-api",
        "credit": "Pay-as-you-go Jobs; requires a positive credit balance",
        "source_url": "https://huggingface.co/docs/hub/jobs-pricing",
    },
    "skypilot": {
        "label": "SkyPilot multi-cloud",
        "kind": "orchestrator",
        "automation": "cli-api",
        "credit": "Uses your authorized AWS/GCP/RunPod/etc. accounts and grants",
        "source_url": "https://docs.skypilot.co/en/latest/getting-started/installation.html",
    },
    "kaggle": {
        "label": "Kaggle Notebooks",
        "kind": "notebook",
        "automation": "export-only",
        "credit": "Free weekly GPU quota; quota and hardware depend on availability",
        "source_url": "https://www.kaggle.com/docs/efficient-gpu-usage",
    },
    "aws_research": {
        "label": "AWS Research Credits",
        "kind": "grant",
        "automation": "via-skypilot",
        "credit": "Eligible graduate/postgraduate/PhD students may request up to $5k",
        "source_url": "https://aws.amazon.com/government-education/research-and-technical-computing/cloud-credit-for-research/faqs/",
    },
}


def _command(name: str) -> str | None:
    discovered = shutil.which(name)
    if discovered:
        return discovered
    suffix = ".exe" if os.name == "nt" else ""
    # Keep the lexical virtualenv path. Resolving its Python symlink points at
    # /usr/bin and hides console scripts installed beside the venv interpreter.
    sibling = Path(sys.executable).parent / f"{name}{suffix}"
    return str(sibling) if sibling.is_file() else None


def _modal_profiles(path: Path) -> list[str]:
    """Return profile names without ever parsing or returning credential values."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    names = re.findall(r"(?m)^\s*\[([^]\r\n]+)]\s*$", text)
    return sorted({name.strip() for name in names if name.strip() not in {"settings"}})


def _modal_cli_profiles(command: str | None, fallback_path: Path) -> tuple[list[str], str | None]:
    """Read safe profile aliases through the supported CLI, with TOML fallback."""

    if not command:
        return _modal_profiles(fallback_path), None
    try:
        listed = subprocess.run(
            [command, "profile", "list", "--json"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        current = subprocess.run(
            [command, "profile", "current"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        payload = json.loads(listed.stdout) if listed.returncode == 0 else None
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return _modal_profiles(fallback_path), None
    names: set[str] = set()
    rows = payload.get("profiles", payload) if isinstance(payload, dict) else payload
    if isinstance(rows, dict):
        names.update(str(key) for key in rows)
    elif isinstance(rows, list):
        for row in rows:
            value = row.get("name") if isinstance(row, dict) else row
            if value and SAFE_PROFILE.fullmatch(str(value)):
                names.add(str(value))
    if not names:
        names.update(_modal_profiles(fallback_path))
    active = current.stdout.strip() if current.returncode == 0 else None
    return sorted(name for name in names if SAFE_PROFILE.fullmatch(name)), active or None


def _declared_modal_profiles() -> list[str]:
    path = Path(__file__).resolve().parents[1] / "configs" / "providers" / "modal_boost.yaml"
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except OSError:
        return []
    return sorted(
        str(alias)
        for alias in (payload.get("profiles") or {})
        if SAFE_PROFILE.fullmatch(str(alias))
    )


SAFE_PROFILE = re.compile(r"^[A-Za-z0-9_.-]+$")


def _gcloud_profiles(command: str | None) -> tuple[list[str], bool]:
    """Return configuration aliases and auth presence, never account names or tokens."""
    if not command:
        return [], False
    try:
        configurations = subprocess.run(
            [command, "config", "configurations", "list", "--format=value(name)"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        auth = subprocess.run(
            [command, "auth", "list", "--filter=status:ACTIVE", "--format=value(status)"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return [], False
    profiles = sorted(
        {
            value.strip()
            for value in configurations.stdout.splitlines()
            if value.strip() and SAFE_PROFILE.fullmatch(value.strip())
        }
    )
    return profiles, bool(auth.returncode == 0 and auth.stdout.strip())


def provider_status(provider_settings: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Build a secret-safe provider readiness report for Studio.

    Credentials stay in provider-native stores. Only command presence, credential-store
    presence, and non-secret profile aliases cross the Studio API boundary.
    """
    config = provider_settings or {}
    explicit_modal_path = os.environ.get("MODAL_CONFIG_PATH") or config.get(
        "modal_config_path"
    )
    modal_path = Path(explicit_modal_path or (Path.home() / ".modal.toml")).expanduser()
    modal_command = _command("modal")
    if explicit_modal_path:
        # An explicit store is an isolation boundary (tests, portable installs,
        # or operator-selected credentials). Never mix it with the global CLI
        # store merely because the Modal executable is installed.
        modal_profiles, modal_active = _modal_profiles(modal_path), None
    else:
        modal_profiles, modal_active = _modal_cli_profiles(modal_command, modal_path)
    declared_modal_profiles = _declared_modal_profiles()
    gcloud_command = _command("gcloud")
    gcloud_profiles, gcloud_authenticated = _gcloud_profiles(gcloud_command)
    if not gcloud_profiles:
        gcloud_profiles = sorted(
            {
                str(value)
                for value in config.get("gcp_profiles", [])
                if SAFE_PROFILE.fullmatch(str(value))
            }
        )
    hf_token_present = bool(os.environ.get("HF_TOKEN")) or any(
        path.is_file()
        for path in (
            Path.home() / ".cache" / "huggingface" / "token",
            Path.home() / ".huggingface" / "token",
        )
    )

    probes: dict[str, dict[str, Any]] = {
        "local": {
            "installed": bool(_command("nvidia-smi")),
            "authenticated": True,
            "profiles": ["local"],
        },
        "modal": {
            "installed": bool(modal_command),
            "authenticated": bool(modal_profiles),
            "profiles": modal_profiles,
            "declared_profiles": declared_modal_profiles,
            "profile_status": [
                {"alias": alias, "authenticated": alias in modal_profiles}
                for alias in declared_modal_profiles
            ],
            "active_profile": os.environ.get("MODAL_PROFILE") or modal_active,
            "credential_store": str(modal_path) if modal_path.is_file() else None,
        },
        "gcp": {
            "installed": bool(gcloud_command),
            "authenticated": gcloud_authenticated,
            "profiles": gcloud_profiles,
            "active_profile": os.environ.get("CLOUDSDK_ACTIVE_CONFIG_NAME"),
            "credential_store": None,
            "blockers": [
                "activate_paid_billing_for_gpu",
                "request_regional_gpu_quota",
            ],
        },
        "lightning": {
            "installed": bool(_command("lightning")),
            "authenticated": bool(config.get("lightning_authenticated", False)),
            "profiles": list(config.get("lightning_profiles", [])),
        },
        "huggingface_jobs": {
            "installed": bool(_command("hf")),
            "authenticated": hf_token_present,
            "profiles": list(config.get("huggingface_namespaces", [])),
        },
        "skypilot": {
            "installed": bool(_command("sky")),
            "authenticated": bool(config.get("skypilot_configured", False)),
            "profiles": list(config.get("skypilot_workspaces", [])),
        },
        "kaggle": {
            "installed": bool(_command("kaggle")),
            "authenticated": (Path.home() / ".kaggle" / "kaggle.json").is_file(),
            "profiles": [],
        },
        "aws_research": {
            "installed": bool(_command("aws")),
            "authenticated": (Path.home() / ".aws" / "credentials").is_file(),
            "profiles": list(config.get("aws_profiles", [])),
        },
    }

    rows: list[dict[str, Any]] = []
    for provider_id, metadata in PROVIDER_CATALOG.items():
        probe = probes[provider_id]
        rows.append(
            {
                "id": provider_id,
                **metadata,
                **probe,
                "ready": bool(probe["installed"] and probe["authenticated"]),
                "researched_at": RESEARCHED_AT,
            }
        )
    return rows
