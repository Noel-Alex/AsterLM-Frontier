#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from asterlm import AsterConfig, AsterLM, TrainConfig
from asterlm.artifacts import atomic_write_json
from asterlm.optim import build_hybrid_optimizer
from asterlm.optim.muon import Muon
from asterlm.source_provenance import assert_expected_checkout_source


def _other_compute_processes() -> list[dict[str, Any]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        return []
    processes = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 2)]
        if len(parts) != 3:
            continue
        try:
            pid = int(parts[0])
            memory_mib = float(parts[2])
        except ValueError:
            continue
        if pid != os.getpid():
            processes.append(
                {"pid": pid, "process_name": parts[1], "used_memory_mib": memory_mib}
            )
    return processes


def muon_topology(
    model_config: AsterConfig,
    *,
    per_head: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Extract the exact Muon parameter-group geometry without allocating state."""

    model = AsterLM(model_config)
    optimizer = build_hybrid_optimizer(
        model,
        TrainConfig(
            device="cpu",
            max_steps=1,
            optimizer="muon_adamw",
            muon_per_head=per_head,
        ),
    )
    if optimizer.muon is None:
        raise RuntimeError("Model produced no Muon parameter partition")
    groups = [
        {
            "shapes": [tuple(parameter.shape) for parameter in group["params"]],
            "split_count": int(group.get("split_count", 1)),
            "split_axis": int(group.get("split_axis", 0)),
        }
        for group in optimizer.muon.param_groups
    ]
    shape_counts: dict[str, int] = {}
    matrix_count = 0
    parameter_count = 0
    for group in groups:
        split_count = int(group["split_count"])
        for shape in group["shapes"]:
            key = f"{shape[0]}x{shape[1]}@split{split_count}"
            shape_counts[key] = shape_counts.get(key, 0) + 1
            matrix_count += split_count
            parameter_count += shape[0] * shape[1]
    summary = {
        "parameter_tensor_count": sum(len(group["shapes"]) for group in groups),
        "orthogonalized_matrix_count": matrix_count,
        "parameter_count": parameter_count,
        "shape_counts": dict(sorted(shape_counts.items())),
    }
    del optimizer, model
    gc.collect()
    return groups, summary


def _build_synthetic_optimizer(
    topology: list[dict[str, Any]],
    *,
    device: torch.device,
    dtype: torch.dtype,
    megabatch: bool,
    workspace_gib: float,
    seed: int,
) -> Muon:
    generator = torch.Generator(device=device).manual_seed(seed)
    parameter_groups = []
    for group in topology:
        parameters = []
        for shape in group["shapes"]:
            parameter = torch.nn.Parameter(torch.empty(shape, device=device, dtype=dtype))
            parameter.data.normal_(mean=0.0, std=0.02, generator=generator)
            parameter.grad = torch.empty_like(parameter).normal_(generator=generator)
            parameters.append(parameter)
        parameter_groups.append(
            {
                "params": parameters,
                "split_count": group["split_count"],
                "split_axis": group["split_axis"],
            }
        )
    return Muon(
        parameter_groups,
        lr=0.005,
        momentum=0.95,
        weight_decay=0.1,
        ns_steps=5,
        nesterov=True,
        update_rms=0.2,
        megabatch=megabatch,
        megabatch_max_gib=workspace_gib,
    )


def _benchmark(
    topology: list[dict[str, Any]],
    *,
    label: str,
    megabatch: bool,
    device: torch.device,
    dtype: torch.dtype,
    workspace_gib: float,
    warmup: int,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    optimizer = _build_synthetic_optimizer(
        topology,
        device=device,
        dtype=dtype,
        megabatch=megabatch,
        workspace_gib=workspace_gib,
        seed=seed,
    )
    for _ in range(warmup):
        optimizer.step()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    durations = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        started_wall = time.perf_counter()
        start.record()
        optimizer.step()
        end.record()
        torch.cuda.synchronize(device)
        durations.append(
            {
                "cuda_ms": start.elapsed_time(end),
                "wall_ms": (time.perf_counter() - started_wall) * 1000.0,
            }
        )
    parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    finite = all(bool(torch.isfinite(parameter).all()) for parameter in parameters)
    payload = {
        "label": label,
        "megabatch": megabatch,
        "cuda_ms_median": statistics.median(row["cuda_ms"] for row in durations),
        "cuda_ms_mean": statistics.fmean(row["cuda_ms"] for row in durations),
        "wall_ms_median": statistics.median(row["wall_ms"] for row in durations),
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        "all_parameters_finite": finite,
        "measurements": durations,
    }
    del parameters, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark parameterwise versus mega-batched Muon on model topology"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--workspace-gib", type=float, default=0.5)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--no-per-head", action="store_true")
    parser.add_argument("--allow-busy-gpu", action="store_true")
    args = parser.parse_args()
    if min(args.iterations, args.repetitions) <= 0 or args.warmup < 0:
        raise ValueError("iterations/repetitions must be positive and warmup non-negative")
    if args.workspace_gib <= 0:
        raise ValueError("workspace-gib must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("Muon optimizer benchmark requires CUDA")
    busy = _other_compute_processes()
    if busy and not args.allow_busy_gpu:
        raise RuntimeError(
            "Refusing to contaminate a Muon benchmark while other CUDA processes "
            f"are active: {busy}"
        )

    root = Path(__file__).resolve().parents[1]
    source = assert_expected_checkout_source(root)
    config = AsterConfig.from_yaml(args.model)
    topology, topology_summary = muon_topology(
        config,
        per_head=not args.no_per_head,
    )
    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    trials = []
    for repetition in range(args.repetitions):
        order = (False, True) if repetition % 2 == 0 else (True, False)
        for megabatch in order:
            trials.append(
                _benchmark(
                    topology,
                    label="megabatched" if megabatch else "parameterwise",
                    megabatch=megabatch,
                    device=device,
                    dtype=dtype,
                    workspace_gib=args.workspace_gib,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    seed=args.seed + repetition,
                )
            )
    medians = {
        label: statistics.median(
            trial["cuda_ms_median"] for trial in trials if trial["label"] == label
        )
        for label in ("parameterwise", "megabatched")
    }
    payload = {
        "schema_version": 1,
        "status": "ok",
        "source_provenance": source,
        "model_config": str(args.model.resolve()),
        "system": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
        },
        "protocol": {
            "dtype": args.dtype,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "repetitions": args.repetitions,
            "workspace_gib": args.workspace_gib,
            "per_head": not args.no_per_head,
        },
        "topology": topology_summary,
        "trials": trials,
        "aggregate": {
            "median_cuda_ms": medians,
            "megabatch_speedup": medians["parameterwise"] / medians["megabatched"],
        },
    }
    atomic_write_json(args.output, payload)
    print(payload["aggregate"])


if __name__ == "__main__":
    main()
