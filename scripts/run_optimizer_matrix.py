#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from scripts.run_moe_utilization_matrix import (
        atomic_json,
        balanced_orders,
        median,
        repository_provenance,
        sha256,
        wait_for_idle,
    )
except ModuleNotFoundError:  # Direct execution adds scripts/, not the repository root, to sys.path.
    from run_moe_utilization_matrix import (  # type: ignore[no-redef]
        atomic_json,
        balanced_orders,
        median,
        repository_provenance,
        sha256,
        wait_for_idle,
    )

OPTIMIZERS = {
    "apollo-nl": ("apollo_mini", False),
    "apollo-no-nl": ("apollo_mini", True),
    "muon-adamw": ("muon_adamw", False),
}


def nested_float(payload: dict[str, Any], *keys: str) -> float | None:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return float(current) if current is not None else None


def median_present(values: list[float | None]) -> float | None:
    return median([value for value in values if value is not None])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Balanced optimizer systems matrix for a fixed AsterLM model/workload"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", choices=sorted(OPTIMIZERS), default=list(OPTIMIZERS))
    parser.add_argument("--moe-implementation", choices=["reference", "grouped"], default="grouped")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--sequence", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument("--gpu-sample-interval", type=float, default=0.5)
    parser.add_argument("--cooldown-temperature", type=float, default=72.0)
    parser.add_argument("--idle-utilization", type=float, default=12.0)
    parser.add_argument("--trial-timeout", type=float, default=1800.0)
    args = parser.parse_args()

    variants = list(dict.fromkeys(args.variants))
    orders = balanced_orders(variants)
    args.output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    repository = repository_provenance()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "git_commit": repository.get("commit"),
        "repository": repository,
        "model": {"path": str(args.model), "sha256": sha256(args.model)},
        "train_config": {"path": str(args.train_config), "sha256": sha256(args.train_config)},
        "protocol": {
            "steps": args.steps,
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "sequence": args.sequence,
            "batch": args.batch,
            "accum": args.accum,
            "gpu_sample_interval": args.gpu_sample_interval,
            "idle_utilization_ceiling": args.idle_utilization,
            "order": orders,
            "note": "apollo-no-nl changes optimizer behavior and is a cost control, not a quality candidate.",
        },
        "variants": {
            name: {
                "optimizer": OPTIMIZERS[name][0],
                "apollo_norm_growth_limiter": not OPTIMIZERS[name][1]
                if OPTIMIZERS[name][0].startswith("apollo")
                else None,
            }
            for name in variants
        },
        "trials": records,
    }
    atomic_json(args.output / "matrix.json", manifest)

    for repetition in range(args.repetitions):
        for variant in orders[repetition % len(orders)]:
            optimizer, disable_norm_limiter = OPTIMIZERS[variant]
            trial_name = f"r{repetition + 1}-{variant}"
            result_path = args.output / f"{trial_name}.json"
            log_path = args.output / f"{trial_name}.log"
            idle = wait_for_idle(args.cooldown_temperature, args.idle_utilization)
            command = [
                sys.executable,
                "scripts/profile_training.py",
                "--model",
                str(args.model),
                "--train-config",
                str(args.train_config),
                "--optimizer",
                optimizer,
                "--sequence",
                str(args.sequence),
                "--batch",
                str(args.batch),
                "--accum",
                str(args.accum),
                "--steps",
                str(args.steps),
                "--warmup",
                str(args.warmup),
                "--gpu-sample-interval",
                str(args.gpu_sample_interval),
                "--json",
                str(result_path),
            ]
            if disable_norm_limiter:
                command.append("--apollo-disable-norm-limiter")
            environment = dict(os.environ)
            environment["ASTER_MOE_IMPL"] = args.moe_implementation
            started = time.monotonic()
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=environment,
                    timeout=args.trial_timeout,
                    check=False,
                )
            record: dict[str, Any] = {
                "name": trial_name,
                "variant": variant,
                "repetition": repetition + 1,
                "command": command,
                "optimizer": optimizer,
                "apollo_disable_norm_limiter": disable_norm_limiter,
                "idle_before": idle,
                "seconds": time.monotonic() - started,
                "returncode": completed.returncode,
                "result": str(result_path),
                "log": str(log_path),
            }
            if result_path.is_file():
                payload = json.loads(result_path.read_text(encoding="utf-8"))
                record["status"] = payload.get("status")
                record["summary"] = payload.get("summary")
            else:
                record["status"] = "missing_result"
            records.append(record)
            atomic_json(args.output / "matrix.json", manifest)

    aggregate: dict[str, Any] = {}
    for variant in variants:
        summaries = [
            record.get("summary") or {}
            for record in records
            if record["variant"] == variant and record.get("status") == "ok"
        ]
        aggregate[variant] = {
            "successful_repetitions": len(summaries),
            "median_tokens_per_second": median_present(
                [nested_float(summary, "median_tokens_per_second") for summary in summaries]
            ),
            "median_tokens_per_joule": median_present(
                [nested_float(summary, "median_tokens_per_joule") for summary in summaries]
            ),
            "median_gpu_utilization_percent": median_present(
                [nested_float(summary, "gpu", "median_utilization_gpu") for summary in summaries]
            ),
            "median_peak_allocated_gib": median_present(
                [nested_float(summary, "final_memory", "peak_allocated_gib") for summary in summaries]
            ),
            "median_optimizer_cuda_ms": median_present(
                [
                    nested_float(summary, "median_phase_timing", "cuda_ms", "optimizer")
                    for summary in summaries
                ]
            ),
            "median_clip_cuda_ms": median_present(
                [
                    nested_float(summary, "median_phase_timing", "cuda_ms", "clip_and_finite_gate")
                    for summary in summaries
                ]
            ),
        }
    manifest["completed_utc"] = datetime.now(UTC).isoformat()
    manifest["aggregate"] = aggregate
    atomic_json(args.output / "matrix.json", manifest)
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
