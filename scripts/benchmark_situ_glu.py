#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from asterlm.artifacts import atomic_write_json
from asterlm.kernels.situ_glu import situ_glu, situ_glu_reference
from asterlm.source_provenance import assert_expected_checkout_source


def parse_shape(value: str) -> tuple[int, int]:
    try:
        rows, columns = (int(part) for part in value.lower().split("x"))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"invalid shape {value!r}; expected ROWSxCOLUMNS"
        ) from exc
    if min(rows, columns) <= 0:
        raise argparse.ArgumentTypeError("SiTU shape dimensions must be positive")
    return rows, columns


def measure(function, *, warmup: int, iterations: int) -> dict[str, float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    torch.cuda.synchronize()
    return {
        "cuda_ms": start.elapsed_time(end) / iterations,
        "wall_ms": (time.perf_counter() - wall_start) * 1000.0 / iterations,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="K3 SiTU-GLU CUDA fusion benchmark")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shape", action="append", type=parse_shape, default=[])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the SiTU-GLU benchmark")
    if min(args.warmup, args.iterations) <= 0:
        raise ValueError("warmup and iterations must be positive")

    root = Path(__file__).resolve().parents[1]
    source = assert_expected_checkout_source(root)
    dtype = getattr(torch, args.dtype)
    results = []
    for rows, columns in args.shape or [(4096, 512), (8192, 512), (16384, 512)]:
        torch.manual_seed(29)
        packed = torch.randn(rows, columns * 2, device="cuda", dtype=dtype)
        gate, up = packed.chunk(2, dim=-1)

        reference_gate = gate.detach().clone().requires_grad_(True)
        reference_up = up.detach().clone().requires_grad_(True)
        fused_gate = gate.detach().requires_grad_(True)
        fused_up = up.detach().requires_grad_(True)
        reference = situ_glu_reference(reference_gate, reference_up)
        fused = situ_glu(fused_gate, fused_up)
        gradient = torch.randn_like(reference)
        reference.backward(gradient, retain_graph=True)
        fused.backward(gradient, retain_graph=True)
        relative_l2 = float(
            (fused.float() - reference.float()).norm()
            / reference.float().norm().clamp_min(1e-12)
        )
        gate_gradient_relative_l2 = float(
            (fused_gate.grad.float() - reference_gate.grad.float()).norm()
            / reference_gate.grad.float().norm().clamp_min(1e-12)
        )
        up_gradient_relative_l2 = float(
            (fused_up.grad.float() - reference_up.grad.float()).norm()
            / reference_up.grad.float().norm().clamp_min(1e-12)
        )

        def reference_step(
            reference_gate: torch.Tensor = reference_gate,
            reference_up: torch.Tensor = reference_up,
            gradient: torch.Tensor = gradient,
        ) -> None:
            reference_gate.grad = None
            reference_up.grad = None
            situ_glu_reference(reference_gate, reference_up).backward(gradient)

        def fused_step(
            fused_gate: torch.Tensor = fused_gate,
            fused_up: torch.Tensor = fused_up,
            gradient: torch.Tensor = gradient,
        ) -> None:
            fused_gate.grad = None
            fused_up.grad = None
            situ_glu(fused_gate, fused_up).backward(gradient)

        reference_timing = measure(
            reference_step, warmup=args.warmup, iterations=args.iterations
        )
        fused_timing = measure(
            fused_step, warmup=args.warmup, iterations=args.iterations
        )
        results.append(
            {
                "shape": [rows, columns],
                "parity": {
                    "output_relative_l2": relative_l2,
                    "gate_gradient_relative_l2": gate_gradient_relative_l2,
                    "up_gradient_relative_l2": up_gradient_relative_l2,
                },
                "reference": reference_timing,
                "fused": fused_timing,
                "cuda_speedup": reference_timing["cuda_ms"] / fused_timing["cuda_ms"],
            }
        )

    payload = {
        "schema_version": 1,
        "status": "complete",
        "source_provenance": source,
        "system": {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
        },
        "protocol": {
            "dtype": args.dtype,
            "warmup": args.warmup,
            "iterations": args.iterations,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, payload)
    print(payload)


if __name__ == "__main__":
    main()
