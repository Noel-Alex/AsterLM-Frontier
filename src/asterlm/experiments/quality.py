from __future__ import annotations

import ctypes
import gc
import hashlib
import json
import math
import os
import statistics
from collections.abc import Iterable
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import torch
from torch import nn

from asterlm.config import AsterConfig
from asterlm.model import AsterLM


def _release_initialization_audit_memory() -> None:
    """Return large temporary CPU model allocations before launching CUDA children."""

    gc.collect()
    if os.name != "posix":
        return
    try:
        libc = ctypes.CDLL(None)
        malloc_trim = libc.malloc_trim
        malloc_trim.argtypes = [ctypes.c_size_t]
        malloc_trim.restype = ctypes.c_int
        malloc_trim(0)
    except (AttributeError, OSError):
        # Non-glibc providers still receive Python collection; their execution
        # profiles must prove enough host memory in the normal fit gate.
        return


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    raw = value.view(torch.uint8).numpy()
    digest = hashlib.sha256()
    digest.update(memoryview(raw))
    return digest.hexdigest()


def initialized_projection_fingerprints(model: AsterLM) -> dict[str, dict[str, Any]]:
    """Hash tensors covered by Aster's ordinary projection initialization pass."""

    fingerprints: dict[str, dict[str, Any]] = {}
    for module_name, module in model.named_modules():
        is_projection = (
            isinstance(module, (nn.Linear, nn.Embedding))
            or getattr(module, "_aster_linear", False)
        )
        if not is_projection:
            continue
        for parameter_name, parameter in module.named_parameters(
            recurse=False, remove_duplicate=False
        ):
            name = f"{module_name}.{parameter_name}" if module_name else parameter_name
            fingerprints[name] = {
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype).removeprefix("torch."),
                "numel": parameter.numel(),
                "sha256": _tensor_sha256(parameter),
            }
    return fingerprints


def initialized_parameter_fingerprints(model: AsterLM) -> dict[str, dict[str, Any]]:
    """Hash every unique trainable tensor in an initialized model."""

    return {
        name: {
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype).removeprefix("torch."),
            "numel": parameter.numel(),
            "sha256": _tensor_sha256(parameter),
        }
        for name, parameter in model.named_parameters()
    }


def _fingerprint_set_sha256(fingerprints: dict[str, dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for name in sorted(fingerprints):
        record = fingerprints[name]
        digest.update(name.encode("utf-8"))
        digest.update(json.dumps(record, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def audit_identical_model_initialization(
    variants: Iterable[str],
    config: AsterConfig,
    seed: int,
) -> dict[str, Any]:
    """Cryptographically prove identical arms start from identical full models."""

    ids = list(variants)
    if len(ids) < 2:
        raise ValueError("Full initialization audit requires at least two variants")
    records: dict[str, dict[str, Any]] = {}
    reference: dict[str, dict[str, Any]] | None = None
    reference_id = ids[0]
    for variant_id in ids:
        # Aster's named pass covers ordinary projections. Forking and reseeding also
        # makes specialized recurrence/router tensors reproducible for this exact
        # same-architecture optimizer comparison.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = AsterLM(config, named_initialization_seed=seed)
        current = initialized_parameter_fingerprints(model)
        del model
        _release_initialization_audit_memory()
        if reference is None:
            reference = current
        names = set(reference) | set(current)
        mismatches = sorted(
            name
            for name in names
            if name not in reference
            or name not in current
            or reference[name]["shape"] != current[name]["shape"]
            or reference[name]["sha256"] != current[name]["sha256"]
        )
        records[variant_id] = {
            "tensor_count": len(current),
            "parameter_count": sum(value["numel"] for value in current.values()),
            "fingerprint_sha256": _fingerprint_set_sha256(current),
            "mismatches_vs_reference": mismatches,
        }
    status = (
        "ok"
        if all(not value["mismatches_vs_reference"] for value in records.values())
        else "failed"
    )
    result = {
        "seed": seed,
        "reference": reference_id,
        "scope": "all_unique_named_parameters",
        "status": status,
        "variants": records,
    }
    if status != "ok":
        raise RuntimeError(
            "Full initialization parity failed: "
            + str(
                {
                    name: record["mismatches_vs_reference"][:12]
                    for name, record in records.items()
                    if record["mismatches_vs_reference"]
                }
            )
        )
    return result


def audit_named_initialization(
    candidates: Iterable[tuple[str, AsterConfig]], seed: int
) -> dict[str, Any]:
    """Prove same-named/same-shaped initialized tensors match a reference candidate."""

    items = list(candidates)
    if len(items) < 2:
        raise ValueError("Initialization audit requires at least two candidates")

    reference_id, reference_config = items[0]
    reference_model = AsterLM(
        reference_config,
        named_initialization_seed=seed,
    )
    reference = initialized_projection_fingerprints(reference_model)
    del reference_model
    _release_initialization_audit_memory()

    result: dict[str, Any] = {
        "seed": seed,
        "reference": reference_id,
        "status": "ok",
        "candidates": {},
    }
    for candidate_id, config in items[1:]:
        model = AsterLM(config, named_initialization_seed=seed)
        current = initialized_projection_fingerprints(model)
        del model
        _release_initialization_audit_memory()
        shared = {
            name
            for name, value in reference.items()
            if name in current and value["shape"] == current[name]["shape"]
        }
        mismatches = sorted(
            name for name in shared if reference[name]["sha256"] != current[name]["sha256"]
        )
        result["candidates"][candidate_id] = {
            "shared_tensor_count": len(shared),
            "shared_parameter_count": sum(reference[name]["numel"] for name in shared),
            "mismatches": mismatches,
        }
        if mismatches:
            result["status"] = "failed"
    if result["status"] != "ok":
        details = {
            key: value["mismatches"][:12]
            for key, value in result["candidates"].items()
            if value["mismatches"]
        }
        raise RuntimeError(f"Named initialization parity failed: {details}")
    return result


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in source.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def summarize_quality_run(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    rows = read_jsonl(root / "metrics.jsonl")
    training = [row for row in rows if isinstance(row.get("tokens_per_second"), (int, float))]
    evaluations = [row for row in rows if isinstance(row.get("eval_main_loss"), (int, float))]
    experiment: dict[str, Any] = {}
    if (root / "experiment.json").is_file():
        experiment = json.loads((root / "experiment.json").read_text(encoding="utf-8"))

    latest_eval = max(evaluations, key=lambda row: int(row.get("tokens_seen", 0)), default={})
    throughput = [float(row["tokens_per_second"]) for row in training]
    utilization = [
        float(row["gpu_util_percent"])
        for row in training
        if isinstance(row.get("gpu_util_percent"), (int, float))
    ]
    time_sampled_utilization = next(
        (
            row
            for row in reversed(training)
            if isinstance(row.get("gpu_time_sample_count"), (int, float))
            and float(row["gpu_time_sample_count"]) > 0
        ),
        {},
    )
    peak_vram = [
        float(row["cuda_peak_allocated_gb"])
        for row in training
        if isinstance(row.get("cuda_peak_allocated_gb"), (int, float))
    ]
    wall_seconds = [
        float(row["wall_clock_total_seconds"])
        for row in training
        if isinstance(row.get("wall_clock_total_seconds"), (int, float))
    ]
    training_by_step = sorted(
        training, key=lambda row: (int(row.get("step", 0)), int(row.get("tokens_seen", 0)))
    )
    train_config = (experiment.get("train") or {}).get("config") or {}
    max_grad_norm = train_config.get("max_grad_norm")
    losses = [
        float(row["loss"])
        for row in training_by_step
        if isinstance(row.get("loss"), (int, float)) and math.isfinite(float(row["loss"]))
    ]
    nonfinite_loss_count = sum(
        isinstance(row.get("loss"), (int, float)) and not math.isfinite(float(row["loss"]))
        for row in training_by_step
    )
    grad_norms = [
        float(row.get("grad_norm_pre_clip", row.get("grad_norm_clipped")))
        for row in training_by_step
        if isinstance(row.get("grad_norm_pre_clip", row.get("grad_norm_clipped")), (int, float))
        and math.isfinite(float(row.get("grad_norm_pre_clip", row.get("grad_norm_clipped"))))
    ]
    nonfinite_grad_count = sum(
        row.get("grad_all_finite") == 0
        or (
            isinstance(row.get("grad_norm_pre_clip", row.get("grad_norm_clipped")), (int, float))
            and not math.isfinite(
                float(row.get("grad_norm_pre_clip", row.get("grad_norm_clipped")))
            )
        )
        for row in training_by_step
    )
    clip_events = [
        bool(row.get("grad_was_clipped"))
        if row.get("grad_was_clipped") is not None
        else bool(max_grad_norm is not None and value > float(max_grad_norm))
        for row, value in (
            (row, float(row.get("grad_norm_pre_clip", row.get("grad_norm_clipped"))))
            for row in training_by_step
            if isinstance(row.get("grad_norm_pre_clip", row.get("grad_norm_clipped")), (int, float))
        )
    ]
    diagnostic_rows = [
        row
        for row in training_by_step
        if isinstance(row.get("param_global_rms"), (int, float))
    ]
    parameter_rms_drift = None
    if len(diagnostic_rows) >= 2:
        first_rms = float(diagnostic_rows[0]["param_global_rms"])
        last_rms = float(diagnostic_rows[-1]["param_global_rms"])
        if first_rms:
            parameter_rms_drift = (last_rms - first_rms) / first_rms
    optimizer_fractions = [
        float(row["optimizer_submit_seconds"]) / float(row["window_seconds"])
        for row in training_by_step
        if isinstance(row.get("optimizer_submit_seconds"), (int, float))
        and isinstance(row.get("window_seconds"), (int, float))
        and float(row["window_seconds"]) > 0
    ]
    muon_relative_updates = [
        float(row["muon_relative_update_rms_mean"])
        for row in training_by_step
        if isinstance(row.get("muon_relative_update_rms_mean"), (int, float))
    ]

    def percentile(values: list[float], fraction: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    def curve_point(row: dict[str, Any]) -> dict[str, Any]:
        step = int(row.get("step", 0))
        preceding = [item for item in training_by_step if int(item.get("step", 0)) <= step]
        system = preceding[-1] if preceding else {}
        return {
            "step": row.get("step"),
            "tokens_seen": row.get("tokens_seen"),
            "eval_main_loss": row.get("eval_main_loss"),
            "eval_perplexity": row.get("eval_perplexity"),
            "milestone_tokens": row.get("milestone_tokens"),
            "wall_clock_total_seconds": system.get("wall_clock_total_seconds"),
            "estimated_cumulative_flops": system.get("estimated_cumulative_flops"),
        }
    return {
        "status": experiment.get("status", "missing"),
        "status_reason": experiment.get("status_reason"),
        "run_id": experiment.get("run_id"),
        "tokens_seen": int(experiment.get("completed_tokens", latest_eval.get("tokens_seen", 0))),
        "eval_main_loss": latest_eval.get("eval_main_loss"),
        "eval_perplexity": latest_eval.get("eval_perplexity"),
        "eval_tokens": latest_eval.get("tokens_seen"),
        "median_training_tokens_per_second": statistics.median(throughput) if throughput else None,
        "gpu_utilization_sampling": (
            "continuous_time" if time_sampled_utilization else "step_boundary_fallback"
        ),
        "gpu_utilization_sample_count": (
            int(time_sampled_utilization["gpu_time_sample_count"])
            if time_sampled_utilization
            else len(utilization)
        ),
        "mean_gpu_util_percent": time_sampled_utilization.get(
            "gpu_util_time_mean_percent",
            statistics.fmean(utilization) if utilization else None,
        ),
        "median_gpu_util_percent": time_sampled_utilization.get(
            "gpu_util_time_p50_percent",
            statistics.median(utilization) if utilization else None,
        ),
        "p10_gpu_util_percent": time_sampled_utilization.get(
            "gpu_util_time_p10_percent", percentile(utilization, 0.10)
        ),
        "p90_gpu_util_percent": time_sampled_utilization.get(
            "gpu_util_time_p90_percent", percentile(utilization, 0.90)
        ),
        "peak_vram_gib": max(peak_vram) if peak_vram else None,
        "wall_clock_total_seconds": max(wall_seconds) if wall_seconds else None,
        "run_survived": experiment.get("status") == "ok",
        "training_loss_nonfinite_count": nonfinite_loss_count,
        "gradient_nonfinite_count": nonfinite_grad_count,
        "gradient_norm_mean": statistics.fmean(grad_norms) if grad_norms else None,
        "gradient_norm_p95": percentile(grad_norms, 0.95),
        "gradient_norm_max": max(grad_norms, default=None),
        "gradient_clip_fraction": statistics.fmean(clip_events) if clip_events else None,
        "loss_max_upward_logged_step": max(
            (right - left for left, right in pairwise(losses)),
            default=None,
        ),
        "loss_upward_jump_gt_0_5_count": sum(
            right - left > 0.5 for left, right in pairwise(losses)
        ),
        "diagnostic_snapshot_count": len(diagnostic_rows),
        "parameter_global_rms_relative_drift": parameter_rms_drift,
        "optimizer_wall_fraction_mean": (
            statistics.fmean(optimizer_fractions) if optimizer_fractions else None
        ),
        "muon_relative_update_rms_mean": (
            statistics.fmean(muon_relative_updates) if muon_relative_updates else None
        ),
        "learning_curve": [curve_point(row) for row in evaluations],
    }


def latest_complete_checkpoint(run_dir: str | Path) -> Path | None:
    root = Path(run_dir)
    pointer = root / "latest.txt"
    if pointer.is_file():
        target = Path(pointer.read_text(encoding="utf-8").strip())
        if not target.is_absolute():
            target = root / target
        if (target / "checkpoint_manifest.json").is_file():
            return target
    candidates = sorted(root.glob("checkpoint-*"), reverse=True)
    return next(
        (path for path in candidates if (path / "checkpoint_manifest.json").is_file()),
        None,
    )


def archive_incomplete_quality_run(
    run_dir: str | Path,
    archive_parent: str | Path,
) -> Path:
    """Atomically preserve an interrupted metrics-only attempt before a clean retry.

    A metrics-only architecture campaign intentionally has no optimizer checkpoint to
    resume.  Power loss must not make the whole campaign unusable or silently append a
    second initialization to the first attempt's metric stream.  Moving the complete
    attempt directory keeps its configuration, metrics and experiment record together,
    while allowing the deterministic candidate to restart at the canonical run path.
    """

    source = Path(run_dir)
    if not source.is_dir():
        raise FileNotFoundError(f"Incomplete quality run does not exist: {source}")
    destination_root = Path(archive_parent)
    destination_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    destination = destination_root / f"{source.name}--interrupted-{timestamp}"
    counter = 1
    while destination.exists():
        destination = destination_root / (
            f"{source.name}--interrupted-{timestamp}-{counter}"
        )
        counter += 1
    os.replace(source, destination)
    return destination
