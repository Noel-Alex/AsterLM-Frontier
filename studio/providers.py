from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any


RESEARCHED_AT = "2026-08-10"


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
    sibling = Path(sys.executable).resolve().parent / f"{name}{suffix}"
    return str(sibling) if sibling.is_file() else None


def _modal_profiles(path: Path) -> list[str]:
    """Return profile names without ever parsing or returning credential values."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    names = re.findall(r"(?m)^\s*\[([^]\r\n]+)]\s*$", text)
    return sorted({name.strip() for name in names if name.strip() not in {"settings"}})


def provider_status(provider_settings: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Build a secret-safe provider readiness report for Studio.

    Credentials stay in provider-native stores. Only command presence, credential-store
    presence, and non-secret profile aliases cross the Studio API boundary.
    """
    config = provider_settings or {}
    modal_path = Path(
        os.environ.get("MODAL_CONFIG_PATH")
        or config.get("modal_config_path")
        or (Path.home() / ".modal.toml")
    ).expanduser()
    modal_profiles = _modal_profiles(modal_path)
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
            "installed": bool(_command("modal")),
            "authenticated": bool(modal_profiles),
            "profiles": modal_profiles,
            "active_profile": os.environ.get("MODAL_PROFILE"),
            "credential_store": str(modal_path) if modal_path.is_file() else None,
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
