from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from asterlm.artifacts import atomic_write_json

BENCHMARK_SCHEMA_VERSION = 1

# This is the architecture-selection contract. Keep the names stable so historical
# records can be compared without experiment-specific adapters.
MANDATORY_BENCHMARK_FIELDS = (
    "architecture_id",
    "git_sha",
    "config_hash",
    "backend",
    "gpu",
    "precision",
    "total_params",
    "active_params_estimate",
    "training_tokens",
    "wall_time",
    "tokens_per_second",
    "peak_vram",
    "mean_gpu_util",
    "mean_power",
    "energy_per_token",
    "validation_loss",
    "domain_validation_losses",
    "long_context_scores",
    "benchmark_scores",
    "checkpoint_size",
    "optimizer_memory",
    "estimated_modal_cost",
    "time_to_target_loss",
)

_TEXT_FIELDS = {"architecture_id", "git_sha", "config_hash", "backend", "gpu", "precision"}
_MAPPING_FIELDS = {"domain_validation_losses", "long_context_scores", "benchmark_scores"}
_NONNEGATIVE_FIELDS = {
    "total_params",
    "active_params_estimate",
    "training_tokens",
    "wall_time",
    "tokens_per_second",
    "peak_vram",
    "mean_gpu_util",
    "mean_power",
    "energy_per_token",
    "checkpoint_size",
    "optimizer_memory",
    "estimated_modal_cost",
    "time_to_target_loss",
}


class BenchmarkRecordError(ValueError):
    """Raised when a benchmark record cannot be compared safely."""


def validate_benchmark_record(
    record: Mapping[str, Any], *, require_complete: bool = True
) -> dict[str, Any]:
    """Validate and normalize one architecture benchmark record.

    A run may write a partial record while it is in progress. Promotion and Pareto
    tooling must use ``require_complete=True`` so missing measurements cannot be
    mistaken for zero-cost or zero-memory results.
    """

    missing = [field for field in MANDATORY_BENCHMARK_FIELDS if field not in record]
    if missing:
        raise BenchmarkRecordError(f"Missing mandatory benchmark fields: {', '.join(missing)}")

    normalized = dict(record)
    normalized.setdefault("schema_version", BENCHMARK_SCHEMA_VERSION)
    if normalized["schema_version"] != BENCHMARK_SCHEMA_VERSION:
        raise BenchmarkRecordError(
            f"Unsupported benchmark schema_version={normalized['schema_version']!r}"
        )

    null_fields = [field for field in MANDATORY_BENCHMARK_FIELDS if normalized[field] is None]
    if require_complete and null_fields:
        raise BenchmarkRecordError(
            "Incomplete benchmark record; null measurements: " + ", ".join(null_fields)
        )

    for field in _TEXT_FIELDS:
        value = normalized[field]
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise BenchmarkRecordError(f"{field} must be a non-empty string")

    for field in _MAPPING_FIELDS:
        value = normalized[field]
        if value is not None and not isinstance(value, Mapping):
            raise BenchmarkRecordError(f"{field} must be a mapping")
        if isinstance(value, Mapping):
            normalized[field] = dict(value)

    for field in _NONNEGATIVE_FIELDS | {"validation_loss"}:
        value = normalized[field]
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise BenchmarkRecordError(f"{field} must be numeric")
        if not math.isfinite(float(value)):
            raise BenchmarkRecordError(f"{field} must be finite")
        if field in _NONNEGATIVE_FIELDS and value < 0:
            raise BenchmarkRecordError(f"{field} must be non-negative")

    gpu_util = normalized["mean_gpu_util"]
    if gpu_util is not None and gpu_util > 100:
        raise BenchmarkRecordError("mean_gpu_util must be a percentage in [0, 100]")

    return normalized


def write_benchmark_record(
    path: str | Path, record: Mapping[str, Any], *, require_complete: bool = True
) -> dict[str, Any]:
    normalized = validate_benchmark_record(record, require_complete=require_complete)
    atomic_write_json(Path(path), normalized)
    return normalized


def write_benchmark_csv(path: str | Path, records: list[Mapping[str, Any]]) -> None:
    """Write validated complete records with nested score maps encoded as JSON."""

    normalized = [validate_benchmark_record(record) for record in records]
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["schema_version", *MANDATORY_BENCHMARK_FIELDS]
    extras = sorted({key for row in normalized for key in row if key not in fieldnames})
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*fieldnames, *extras])
        writer.writeheader()
        for row in normalized:
            flattened = dict(row)
            for field in _MAPPING_FIELDS:
                flattened[field] = json.dumps(flattened[field], sort_keys=True, separators=(",", ":"))
            writer.writerow(flattened)
