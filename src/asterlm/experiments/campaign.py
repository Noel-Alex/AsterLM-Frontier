from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from asterlm.artifacts import atomic_write_json, sha256_file
from asterlm.config import AsterConfig

_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_ENV_PATTERN = re.compile(r"^[A-Z_][A-Z0-9_]*$")


@dataclass(frozen=True, slots=True)
class ExecutionVariant:
    variant_id: str
    description: str
    model_overrides: dict[str, Any]
    train_overrides: dict[str, Any]
    environment: dict[str, str]


@dataclass(frozen=True, slots=True)
class ArchitectureCandidate:
    candidate_id: str
    tier: int
    hypothesis: str
    base_model: Path
    model_overrides: dict[str, Any]
    execution_variants: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ArchitectureCampaign:
    source: Path
    schema_version: int
    candidates: tuple[ArchitectureCandidate, ...]
    context_lengths: tuple[int, ...]
    comparison_axes: tuple[str, ...]
    execution_variants: dict[str, ExecutionVariant]


def _within(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Campaign path escapes repository root: {relative}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_architecture_campaign(
    path: str | Path, *, repo_root: str | Path | None = None
) -> ArchitectureCampaign:
    source = Path(path).resolve()
    root = Path(repo_root).resolve() if repo_root is not None else source.parents[2]
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if payload.get("schema_version") != 2:
        raise ValueError("Architecture campaign requires schema_version: 2")

    execution_variants: dict[str, ExecutionVariant] = {}
    for variant_id, raw in (payload.get("execution_variants", {}) or {}).items():
        variant_id = str(variant_id)
        if not _ID_PATTERN.fullmatch(variant_id):
            raise ValueError(f"Invalid execution variant id: {variant_id!r}")
        if not isinstance(raw, dict):
            raise TypeError(f"Execution variant {variant_id} must be a mapping")
        description = str(raw.get("description", "")).strip()
        if not description:
            raise ValueError(f"Execution variant {variant_id} has no description")
        model_overrides = dict(raw.get("model_overrides", {}))
        train_overrides = dict(raw.get("train_overrides", {}))
        raw_environment = dict(raw.get("environment", {}))
        bad_environment = [key for key in raw_environment if not _ENV_PATTERN.fullmatch(str(key))]
        if bad_environment:
            raise ValueError(
                f"Execution variant {variant_id} has invalid environment keys: {bad_environment}"
            )
        environment = {str(key): str(value) for key, value in raw_environment.items()}
        execution_variants[variant_id] = ExecutionVariant(
            variant_id=variant_id,
            description=description,
            model_overrides=model_overrides,
            train_overrides=train_overrides,
            environment=environment,
        )
    if not execution_variants:
        raise ValueError("Campaign must define execution_variants")

    context_lengths = tuple(int(value) for value in payload.get("context_lengths", []))
    if not context_lengths or any(value <= 0 for value in context_lengths):
        raise ValueError("context_lengths must contain positive integers")
    if tuple(sorted(set(context_lengths))) != context_lengths:
        raise ValueError("context_lengths must be unique and increasing")

    comparison_axes = tuple(str(value) for value in payload.get("comparison_axes", []))
    required_axes = {"equal_tokens", "equal_active_flops", "equal_wall_time", "equal_cost"}
    missing_axes = required_axes - set(comparison_axes)
    if missing_axes:
        raise ValueError(f"Campaign is missing comparison axes: {sorted(missing_axes)}")

    candidates: list[ArchitectureCandidate] = []
    seen: set[str] = set()
    for raw in payload.get("candidates", []):
        candidate_id = str(raw.get("id", ""))
        if not _ID_PATTERN.fullmatch(candidate_id):
            raise ValueError(f"Invalid architecture candidate id: {candidate_id!r}")
        if candidate_id in seen:
            raise ValueError(f"Duplicate architecture candidate id: {candidate_id}")
        seen.add(candidate_id)
        tier = int(raw.get("tier", -1))
        if tier not in range(6):
            raise ValueError(f"Candidate {candidate_id} has invalid tier {tier}")
        hypothesis = str(raw.get("hypothesis", "")).strip()
        if not hypothesis:
            raise ValueError(f"Candidate {candidate_id} has no hypothesis")
        base_model = _within(root, str(raw.get("base_model", "")))
        overrides = dict(raw.get("model_overrides", {}))
        base_payload = yaml.safe_load(base_model.read_text(encoding="utf-8")) or {}
        model_payload = dict(base_payload.get("model", base_payload))
        model_payload.update(overrides)
        AsterConfig(**model_payload)  # Fail before spending compute on an invalid materialization.
        candidate_variants = tuple(str(v) for v in raw.get("execution_variants", []))
        if not candidate_variants:
            raise ValueError(f"Candidate {candidate_id} has no execution variants")
        if len(set(candidate_variants)) != len(candidate_variants):
            raise ValueError(f"Candidate {candidate_id} repeats an execution variant")
        unknown_variants = sorted(set(candidate_variants) - set(execution_variants))
        if unknown_variants:
            raise ValueError(
                f"Candidate {candidate_id} references unknown execution variants: {unknown_variants}"
            )
        for variant_id in candidate_variants:
            effective_model = dict(model_payload)
            effective_model.update(execution_variants[variant_id].model_overrides)
            AsterConfig(**effective_model)
        candidates.append(
            ArchitectureCandidate(
                candidate_id=candidate_id,
                tier=tier,
                hypothesis=hypothesis,
                base_model=base_model,
                model_overrides=overrides,
                execution_variants=candidate_variants,
            )
        )

    if not candidates or not any(candidate.tier == 0 for candidate in candidates):
        raise ValueError("Campaign must contain a Tier-0 reference")
    return ArchitectureCampaign(
        source=source,
        schema_version=2,
        candidates=tuple(candidates),
        context_lengths=context_lengths,
        comparison_axes=comparison_axes,
        execution_variants=execution_variants,
    )


def materialize_architecture_campaign(
    campaign: ArchitectureCampaign, output_dir: str | Path
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "campaign_source": str(campaign.source),
        "campaign_source_sha256": sha256_file(campaign.source),
        "context_lengths": list(campaign.context_lengths),
        "comparison_axes": list(campaign.comparison_axes),
        "execution_variants": {
            variant_id: {
                "description": variant.description,
                "model_overrides": variant.model_overrides,
                "train_overrides": variant.train_overrides,
                "environment": variant.environment,
            }
            for variant_id, variant in campaign.execution_variants.items()
        },
        "candidates": {},
    }
    for candidate in campaign.candidates:
        base_payload = yaml.safe_load(candidate.base_model.read_text(encoding="utf-8")) or {}
        model_payload = dict(base_payload.get("model", base_payload))
        model_payload.update(candidate.model_overrides)
        AsterConfig(**model_payload)
        serialized = yaml.safe_dump({"model": model_payload}, sort_keys=False)
        target = output / f"{candidate.candidate_id}.yaml"
        temporary = target.with_suffix(target.suffix + ".partial")
        temporary.write_text(serialized, encoding="utf-8")
        os.replace(temporary, target)
        config_hash = hashlib.sha256(
            json.dumps(model_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        manifest["candidates"][candidate.candidate_id] = {
            "tier": candidate.tier,
            "hypothesis": candidate.hypothesis,
            "base_model": str(candidate.base_model),
            "base_model_sha256": sha256_file(candidate.base_model),
            "model_overrides": candidate.model_overrides,
            "execution_variants": list(candidate.execution_variants),
            "materialized_config": str(target),
            "config_sha256": config_hash,
            "effective_variants": {},
        }
        for variant_id in candidate.execution_variants:
            variant = campaign.execution_variants[variant_id]
            effective_model = dict(model_payload)
            effective_model.update(variant.model_overrides)
            AsterConfig(**effective_model)
            variant_target = output / f"{candidate.candidate_id}--{variant_id}.yaml"
            variant_serialized = yaml.safe_dump({"model": effective_model}, sort_keys=False)
            variant_temporary = variant_target.with_suffix(variant_target.suffix + ".partial")
            variant_temporary.write_text(variant_serialized, encoding="utf-8")
            os.replace(variant_temporary, variant_target)
            variant_hash = hashlib.sha256(
                json.dumps(effective_model, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            manifest["candidates"][candidate.candidate_id]["effective_variants"][variant_id] = {
                "materialized_config": str(variant_target),
                "config_sha256": variant_hash,
                "model_overrides": variant.model_overrides,
                "train_overrides": variant.train_overrides,
                "environment": variant.environment,
            }
    atomic_write_json(output / "campaign-manifest.json", manifest)
    return manifest
