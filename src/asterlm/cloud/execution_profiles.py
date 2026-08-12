from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import yaml


def resolve_gpu_execution_profile(
    train_path: str | Path,
    gpu: str,
    *,
    profiles_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve a reproducible batch geometry for one exact accelerator tier."""

    train_path = Path(train_path)
    profiles_path = Path(profiles_path)
    train_payload = yaml.safe_load(train_path.read_text(encoding="utf-8")) or {}
    profile_payload = yaml.safe_load(profiles_path.read_text(encoding="utf-8")) or {}
    if profile_payload.get("schema_version") != 1:
        raise ValueError("GPU execution profile config requires schema_version: 1")
    train = train_payload.get("train")
    if not isinstance(train, dict):
        raise ValueError("Training config has no train mapping")
    selected_name: str | None = None
    selected: dict[str, Any] | None = None
    for name, candidate in (profile_payload.get("profiles") or {}).items():
        aliases = {str(value).lower() for value in candidate.get("aliases", [])}
        if gpu.lower() == str(name).lower() or gpu.lower() in aliases:
            selected_name = str(name)
            selected = candidate
            break
    if selected is None or selected_name is None:
        raise ValueError(f"No GPU execution profile is declared for {gpu!r}")

    sequence = int(train["sequence_length"])
    micro_map = {int(key): int(value) for key, value in selected["micro_batch_by_context"].items()}
    if sequence not in micro_map:
        raise ValueError(f"GPU profile {selected_name} has no {sequence}-token context recipe")
    micro_batch = micro_map[sequence]
    target_tokens = int(profile_payload["target_effective_batch_tokens"])
    accumulation = max(1, math.ceil(target_tokens / (sequence * micro_batch)))
    effective_tokens = sequence * micro_batch * accumulation
    train.update(
        {
            "execution_backend": "aster_local",
            "execution_autotune": False,
            "moe_implementation": "cutlass",
            "micro_batch_size": micro_batch,
            "gradient_accumulation_steps": accumulation,
            "muon_megabatch": True,
            "muon_megabatch_max_gib": float(selected["muon_megabatch_max_gib"]),
            "activation_offload": False,
        }
    )
    identity = {
        "schema_version": 1,
        "gpu_requested": gpu,
        "profile": selected_name,
        "sequence_length": sequence,
        "micro_batch_size": micro_batch,
        "gradient_accumulation_steps": accumulation,
        "effective_batch_tokens": effective_tokens,
        "target_effective_batch_tokens": target_tokens,
        "precision_backend": train.get("precision_backend"),
        "dtype": train.get("dtype"),
        "moe_implementation": train.get("moe_implementation"),
        "source_train_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
        "profiles_sha256": hashlib.sha256(profiles_path.read_bytes()).hexdigest(),
    }
    return train_payload, identity
