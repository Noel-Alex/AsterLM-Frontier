from __future__ import annotations

import hashlib
import json
import os
import statistics
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn

from asterlm.config import AsterConfig
from asterlm.model import AsterLM


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
        "mean_gpu_util_percent": statistics.fmean(utilization) if utilization else None,
        "peak_vram_gib": max(peak_vram) if peak_vram else None,
        "wall_clock_total_seconds": max(wall_seconds) if wall_seconds else None,
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
