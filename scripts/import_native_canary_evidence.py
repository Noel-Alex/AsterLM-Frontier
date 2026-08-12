#!/usr/bin/env python3
"""Validate and archive a bounded, source-pinned native-context laptop canary."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = ROOT / "runs/promotion-canary/k3-native8k-muon-5b59fe9-20260812/matrix.json"
DEFAULT_GATES = ROOT / "configs/experiments/promotion_gates.yaml"
DEFAULT_OUTPUT = ROOT / "docs/promotion-evidence"
REQUIRED_TRIALS = 3
MIN_MEDIAN_GPU_UTILIZATION = 90.0
MAX_PEAK_ALLOCATED_GIB = 11.25
CERTIFIED_GATES = (
    "repeated_warm_throughput",
    "gpu_utilization_root_cause",
    "vram",
    "energy_and_power",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def repo_relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.strip().lower()


def require_clean_checkout() -> None:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        raise RuntimeError("Native canary evidence import must start from a clean checkout")


def validate_matrix(matrix_path: Path, matrix: dict[str, Any]) -> dict[str, Any]:
    if matrix.get("schema_version") != 2:
        raise ValueError("Unexpected canary matrix schema")
    source = matrix.get("source_provenance") or {}
    execution = matrix.get("execution_checkout") or {}
    repository = matrix.get("repository") or {}
    source_commit = str(source.get("git_commit") or "")
    if not source_commit or source.get("dirty") is not False:
        raise ValueError("Canary source was not clean and pinned")
    if execution.get("commit") != source_commit or execution.get("dirty") is not False:
        raise ValueError("Canary execution checkout does not match the source pin")
    if repository.get("commit") != source_commit or repository.get("dirty") is not False:
        raise ValueError("Canary launch checkout was dirty or mismatched")

    protocol = matrix.get("protocol") or {}
    expected = {
        "sequence": 8192,
        "batch": 1,
        "accum": 16,
        "steps": 2,
        "warmup": 1,
        "repetitions": REQUIRED_TRIALS,
        "optimizer_override": "muon_adamw",
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"Native canary protocol mismatch for {key}: {protocol.get(key)!r}")
    model = (matrix.get("models") or {}).get("k3-final") or {}
    if model.get("moe_implementation") != "cutlass":
        raise ValueError("Native canary did not use the frozen CUTLASS MoE backend")

    trials = matrix.get("trials") or []
    if len(trials) != REQUIRED_TRIALS or any(row.get("status") != "ok" for row in trials):
        raise ValueError("Native canary did not complete all repetitions")
    rows = []
    for trial in trials:
        result_path = Path(str(trial["result"]))
        if not result_path.is_file():
            result_path = matrix_path.parent / result_path.name
        payload = load_json(result_path)
        summary = payload.get("summary") or {}
        system = payload.get("system") or {}
        resolved = payload.get("resolved_train") or {}
        if payload.get("status") != "ok" or system.get("git_commit") != source_commit:
            raise ValueError(f"Invalid source-pinned trial: {trial.get('name')}")
        if system.get("git_dirty") is not False:
            raise ValueError("Trial ran from dirty source")
        if payload.get("moe_implementation") != "cutlass":
            raise ValueError("Trial used the wrong MoE backend")
        if resolved.get("optimizer") != "muon_adamw" or resolved.get("sequence_length") != 8192:
            raise ValueError("Trial does not match the frozen stage-1 execution recipe")
        gpu = summary.get("gpu") or {}
        memory = summary.get("final_memory") or {}
        if int(gpu.get("measured_sample_count") or 0) < 2:
            raise ValueError("Trial has insufficient measured GPU samples")
        rows.append(
            {
                "name": trial["name"],
                "result_sha256": sha256(result_path),
                "median_tokens_per_second": float(summary["median_tokens_per_second"]),
                "mean_gpu_utilization": float(gpu["mean_utilization_gpu"]),
                "median_gpu_utilization": float(gpu["median_utilization_gpu"]),
                "p10_gpu_utilization": float(gpu["p10_utilization_gpu"]),
                "peak_allocated_gib": float(memory["peak_allocated_gib"]),
                "peak_reserved_gib": float(memory["peak_reserved_gib"]),
                "estimated_gpu_energy_kwh": float(summary["estimated_gpu_energy_kwh"]),
                "estimated_gpu_joules_per_token": float(summary["estimated_gpu_joules_per_token"]),
                "host_metadata_syncs_per_forward": (
                    float((summary.get("moe_execution") or {})["moe_backend_host_metadata_syncs"])
                    / float((summary.get("moe_execution") or {})["moe_backend_forward_calls"])
                ),
            }
        )
    aggregate = (matrix.get("aggregate") or {}).get("k3-final") or {}
    if aggregate.get("successful_repetitions") != REQUIRED_TRIALS:
        raise ValueError("Canary aggregate is incomplete")
    if float(aggregate["median_gpu_utilization"]) < MIN_MEDIAN_GPU_UTILIZATION:
        raise ValueError("Native 8K GPU utilization remains below the promotion floor")
    if float(aggregate["median_peak_allocated_gib"]) > MAX_PEAK_ALLOCATED_GIB:
        raise ValueError("Native 8K peak allocation exceeds the safety floor")
    if any(row["p10_gpu_utilization"] < 85.0 for row in rows):
        raise ValueError("A repetition contains sustained low-utilization samples")

    return {
        "source_execution_commit": source_commit,
        "hardware": "NVIDIA GeForce RTX 4080 Laptop GPU (Ada SM89, 12 GiB)",
        "protocol": expected,
        "trials": rows,
        "aggregate": aggregate,
        "thresholds": {
            "minimum_median_gpu_utilization_percent": MIN_MEDIAN_GPU_UTILIZATION,
            "maximum_peak_allocated_gib": MAX_PEAK_ALLOCATED_GIB,
            "minimum_trial_p10_gpu_utilization_percent": 85.0,
        },
        "diagnostics": {
            "remaining_host_metadata_syncs_per_forward": float(
                aggregate["median_moe_host_metadata_syncs_per_forward"]
            ),
            "interpretation": (
                "The native 8K workload sustains high GPU occupancy; one host metadata sync per "
                "MoE forward remains a measurable optimization target but no longer starves the GPU."
            ),
        },
    }


def update_ledger(path: Path, records: dict[str, dict[str, str]]) -> None:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    seen: set[str] = set()
    for gate in payload.get("gates", []):
        gate_id = str(gate.get("id") or "")
        if gate_id in records:
            gate["status"] = "passed"
            gate["evidence"] = [records[gate_id]]
            seen.add(gate_id)
    if seen != set(records):
        raise RuntimeError(f"Promotion ledger missing gates: {sorted(set(records) - seen)}")
    atomic_text(path, yaml.safe_dump(payload, sort_keys=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--gates", type=Path, default=DEFAULT_GATES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    require_clean_checkout()
    importer_commit = git_commit()
    matrix = load_json(args.matrix)
    assertions = validate_matrix(args.matrix, matrix)

    output = args.output / importer_commit[:12] / "native-8k-canary"
    output.mkdir(parents=True, exist_ok=True)
    matrix_copy = output / "matrix.json"
    shutil.copyfile(args.matrix, matrix_copy)
    result = output / "native-8k-canary-result.json"
    atomic_text(
        result,
        json.dumps(
            {
                "schema_version": 1,
                "status": "passed",
                "imported_at_utc": datetime.now(timezone.utc).isoformat(),
                "importer_git_commit": importer_commit,
                "source_matrix": {"path": repo_relative(matrix_copy), "sha256": sha256(matrix_copy)},
                "assertions": assertions,
                "certified_gates": list(CERTIFIED_GATES),
                "not_certified": [
                    "equal_wall_clock",
                    "long_context_retrieval",
                    "interrupted_laptop_recovery",
                    "laptop_inference",
                ],
            },
            indent=2,
        )
        + "\n",
    )
    artifacts = [
        {"path": repo_relative(matrix_copy), "sha256": sha256(matrix_copy)},
        {"path": repo_relative(result), "sha256": sha256(result)},
    ]
    records: dict[str, dict[str, str]] = {}
    for gate_id in CERTIFIED_GATES:
        proof = output / f"{gate_id}-proof.json"
        atomic_text(
            proof,
            json.dumps(
                {
                    "schema_version": 1,
                    "gate_id": gate_id,
                    "status": "passed",
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "evaluator": {
                        "name": "asterlm-native-canary-importer",
                        "version": "1",
                        "git_commit": importer_commit,
                    },
                    "experiment_ids": [
                        "k3-native8k-muon-5b59fe9-20260812",
                        *[row["name"] for row in assertions["trials"]],
                    ],
                    "artifacts": artifacts,
                },
                indent=2,
            )
            + "\n",
        )
        records[gate_id] = {"path": repo_relative(proof), "sha256": sha256(proof)}
    update_ledger(args.gates, records)

    from asterlm.experiments import evaluate_promotion_gates

    print(
        json.dumps(
            {"assertions": assertions, "promotion": evaluate_promotion_gates(args.gates).manifest()},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
