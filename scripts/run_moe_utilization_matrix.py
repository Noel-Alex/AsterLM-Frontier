#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

VARIANTS = {
    "dense-all-mla": ("model-screen-dense-all-mla.yaml", "reference"),
    "dense-kda3": ("model-screen-dense-kda3.yaml", "reference"),
    "grouped-moe-kda3": ("model-screen-moe-grouped-kda3.yaml", "grouped"),
    "latent-moe-kda3": ("model-screen-latentmoe-kda3.yaml", "grouped"),
}


def balanced_orders(variants: list[str]) -> list[list[str]]:
    """Rotate trial order across repetitions to reduce thermal/order bias."""
    return [variants[offset:] + variants[:offset] for offset in range(len(variants))]


def sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes())
    return digest.hexdigest()


def repository_provenance(root: Path | None = None) -> dict[str, Any]:
    """Capture the exact tracked worktree state used to launch an experiment."""

    root = (root or Path.cwd()).resolve()

    def git(*arguments: str, text: bool = True) -> subprocess.CompletedProcess[Any]:
        return subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=False,
            capture_output=True,
            text=text,
        )

    commit_result = git("rev-parse", "HEAD")
    status_result = git("status", "--porcelain=v1", "--untracked-files=all")
    diff_result = git("diff", "--binary", "HEAD", "--", text=False)
    if commit_result.returncode or status_result.returncode or diff_result.returncode:
        errors = [
            output.strip()
            for output in (commit_result.stderr, status_result.stderr)
            if output.strip()
        ]
        if diff_result.stderr:
            errors.append(diff_result.stderr.decode(errors="replace").strip())
        return {
            "available": False,
            "root": str(root),
            "error": "; ".join(errors) or "git provenance commands failed",
        }

    status_lines = status_result.stdout.splitlines()
    tracked_diff = diff_result.stdout
    return {
        "available": True,
        "root": str(root),
        "commit": commit_result.stdout.strip(),
        "dirty": bool(status_lines),
        "status_porcelain": status_lines,
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        "tracked_diff_bytes": len(tracked_diff),
        "note": "Hash covers staged and unstaged tracked changes; status also records untracked paths.",
    }


def gpu_state() -> dict[str, float] | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,temperature.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        values = [float(value.strip()) for value in output.splitlines()[0].split(",")]
        return dict(zip(("utilization_gpu", "temperature_gpu", "memory_used_mib"), values, strict=True))
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def wait_for_idle(
    max_temperature: float,
    max_utilization: float,
    timeout: float = 300.0,
) -> dict[str, float]:
    deadline = time.monotonic() + timeout
    consecutive = 0
    last: dict[str, float] | None = None
    while time.monotonic() < deadline:
        last = gpu_state()
        if (
            last is not None
            and last["utilization_gpu"] <= max_utilization
            and last["temperature_gpu"] <= max_temperature
            and last["memory_used_mib"] <= 1024
        ):
            consecutive += 1
            if consecutive >= 3:
                return last
        else:
            consecutive = 0
        time.sleep(1)
    raise TimeoutError(f"GPU did not become idle before trial; last state: {last}")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Matched dense/MoE utilization follow-up")
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--sequence", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--accum", type=int, default=16)
    parser.add_argument("--gpu-sample-interval", type=float, default=0.5)
    parser.add_argument("--cooldown-temperature", type=float, default=72.0)
    parser.add_argument("--idle-utilization", type=float, default=12.0)
    parser.add_argument("--trial-timeout", type=float, default=1800.0)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Subset of registered variant names to benchmark.",
    )
    parser.add_argument(
        "--variant-spec",
        action="append",
        default=[],
        metavar="NAME=FILENAME=IMPLEMENTATION",
        help="Register an experiment-local model config without editing this script.",
    )
    args = parser.parse_args()

    variants = dict(VARIANTS)
    custom_names: list[str] = []
    for raw_spec in args.variant_spec:
        try:
            name, filename, implementation = raw_spec.split("=", 2)
        except ValueError as exc:
            raise SystemExit(f"Invalid --variant-spec {raw_spec!r}; expected NAME=FILENAME=IMPLEMENTATION") from exc
        if not name or not filename or implementation not in {
            "reference",
            "grouped",
            "cutlass",
        }:
            raise SystemExit(f"Invalid --variant-spec {raw_spec!r}")
        variants[name] = (filename, implementation)
        custom_names.append(name)

    requested = args.variants if args.variants is not None else (custom_names or list(VARIANTS))
    selected_variants = list(dict.fromkeys(requested))
    unknown = [name for name in selected_variants if name not in variants]
    if unknown:
        raise SystemExit(
            "Unknown variant(s): " + ", ".join(unknown) + ". Available: " + ", ".join(sorted(variants))
        )
    orders = balanced_orders(selected_variants)

    args.output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    repository = repository_provenance()
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "created_utc": datetime.now(UTC).isoformat(),
        "git_commit": repository.get("commit"),
        "repository": repository,
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
        },
        "external_gpu_processes_at_start": subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ).stdout.strip(),
        "train_config": {"path": str(args.train_config), "sha256": sha256(args.train_config)},
        "models": {},
        "trials": records,
    }
    for variant in selected_variants:
        filename, implementation = variants[variant]
        model_path = args.config_root / filename
        manifest["models"][variant] = {
            "path": str(model_path),
            "sha256": sha256(model_path),
            "moe_implementation": implementation,
        }
    atomic_json(args.output / "matrix.json", manifest)

    for repetition in range(args.repetitions):
        order = orders[repetition % len(orders)]
        for variant in order:
            filename, implementation = variants[variant]
            trial_name = f"r{repetition + 1}-{variant}"
            result_path = args.output / f"{trial_name}.json"
            log_path = args.output / f"{trial_name}.log"
            idle = wait_for_idle(args.cooldown_temperature, args.idle_utilization)
            command = [
                sys.executable,
                "scripts/profile_training.py",
                "--model",
                str(args.config_root / filename),
                "--train-config",
                str(args.train_config),
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
            environment = dict(os.environ)
            environment["ASTER_MOE_IMPL"] = implementation
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
                "moe_implementation": implementation,
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
    for variant in selected_variants:
        successful = [
            record for record in records
            if record["variant"] == variant and record.get("status") == "ok"
        ]
        summaries = [record.get("summary") or {} for record in successful]
        aggregate[variant] = {
            "successful_repetitions": len(successful),
            "median_tokens_per_second": median(
                [float(summary["median_tokens_per_second"]) for summary in summaries]
            ),
            "median_tokens_per_joule": median(
                [float(summary["median_tokens_per_joule"]) for summary in summaries]
            ),
            "median_gpu_utilization": median(
                [float(summary["gpu"]["median_utilization_gpu"]) for summary in summaries]
            ),
            "median_gpu_power_w": median(
                [float(summary["gpu"]["median_power_draw"]) for summary in summaries]
            ),
            "median_peak_allocated_gib": median(
                [float(summary["final_memory"]["peak_allocated_gib"]) for summary in summaries]
            ),
        }
    manifest["completed_utc"] = datetime.now(UTC).isoformat()
    manifest["aggregate"] = aggregate
    atomic_json(args.output / "matrix.json", manifest)
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
