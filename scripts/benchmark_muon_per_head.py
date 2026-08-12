#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from asterlm.artifacts import atomic_write_json
from asterlm.optim.muon import (
    zeropower_via_newton_schulz5,
    zeropower_via_newton_schulz5_batched,
)
from asterlm.source_provenance import assert_expected_checkout_source


def parse_shape(value: str) -> tuple[int, int, int]:
    try:
        blocks, rows, columns = (int(part) for part in value.lower().split("x"))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"invalid shape {value!r}; expected BLOCKSxROWSxCOLUMNS"
        ) from exc
    if min(blocks, rows, columns) <= 0:
        raise argparse.ArgumentTypeError("Muon shape dimensions must be positive")
    return blocks, rows, columns


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


def _time_cuda(function, *, warmup: int, iterations: int) -> dict[str, float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    started_wall = time.perf_counter()
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    torch.cuda.synchronize()
    return {
        "cuda_ms_per_call": start.elapsed_time(end) / iterations,
        "wall_ms_per_call": (time.perf_counter() - started_wall) * 1000.0 / iterations,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parity and launch-fragmentation benchmark for K3 per-head Muon"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--shape",
        action="append",
        type=parse_shape,
        default=[],
        help="Per-head batch geometry BLOCKSxROWSxCOLUMNS; repeat for multiple projections.",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--ns-steps", type=int, default=5)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--allow-busy-gpu", action="store_true")
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations <= 0 or args.ns_steps <= 0:
        raise ValueError("warmup must be non-negative; iterations and ns-steps must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("Muon CUDA benchmark requires a CUDA GPU")
    busy = _other_compute_processes()
    if busy and not args.allow_busy_gpu:
        raise RuntimeError(
            "Refusing to contaminate a Muon benchmark while other CUDA compute processes "
            f"are active: {busy}"
        )

    root = Path(__file__).resolve().parents[1]
    source = assert_expected_checkout_source(root)
    shapes = args.shape or [(6, 128, 768), (12, 64, 768), (12, 64, 64)]
    dtype = getattr(torch, args.dtype)
    device = torch.device("cuda")
    results = []
    for blocks, rows, columns in shapes:
        matrices = torch.randn(
            blocks, rows, columns, device=device, dtype=dtype
        )

        def independent(current_matrices=matrices, steps=args.ns_steps):
            return torch.stack(
                [
                    zeropower_via_newton_schulz5(matrix, steps=steps)
                    for matrix in current_matrices
                ]
            )

        def batched(current_matrices=matrices, steps=args.ns_steps):
            return zeropower_via_newton_schulz5_batched(
                current_matrices, steps=steps
            )

        expected = independent()
        actual = batched()
        difference = (actual.float() - expected.float()).flatten()
        denominator = expected.float().flatten().norm().clamp_min(1e-12)
        parity = {
            "max_abs": float(difference.abs().max()),
            "relative_l2": float(difference.norm() / denominator),
            "all_finite": bool(torch.isfinite(actual).all()),
        }
        independent_timing = _time_cuda(
            independent, warmup=args.warmup, iterations=args.iterations
        )
        batched_timing = _time_cuda(
            batched, warmup=args.warmup, iterations=args.iterations
        )
        speedup = (
            independent_timing["cuda_ms_per_call"]
            / batched_timing["cuda_ms_per_call"]
        )
        results.append(
            {
                "shape": [blocks, rows, columns],
                "matrices_per_call": blocks,
                "parity": parity,
                "independent": independent_timing,
                "batched": batched_timing,
                "cuda_speedup": speedup,
            }
        )
        del matrices, expected, actual, independent, batched

    payload = {
        "schema_version": 1,
        "status": "ok",
        "source_provenance": source,
        "system": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
        },
        "protocol": {
            "dtype": args.dtype,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "newton_schulz_steps": args.ns_steps,
            "busy_gpu_override": args.allow_busy_gpu,
        },
        "results": results,
    }
    atomic_write_json(args.output, payload)
    print(payload)


if __name__ == "__main__":
    main()
