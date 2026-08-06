#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import torch

from asterlm.training.telemetry import SystemSampler, static_system_manifest


def median(values: list[float]) -> float:
    values = sorted(values)
    return values[len(values) // 2]


def timed_cuda(fn: Callable[[], None], repeats: int, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    durations: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        durations.append(start.elapsed_time(end) / 1000.0)
    return median(durations)


def benchmark_linear(device: torch.device, dtype: torch.dtype, repeats: int) -> dict[str, Any]:
    # Sizes are deliberately aligned to 16 for Ada FP8 and representative of the
    # 1152-wide frontier model's large FFN projections.
    m, k, n = 8192, 1152, 4608
    x = torch.randn(m, k, device=device, dtype=dtype)
    layer = torch.nn.Linear(k, n, bias=False, device=device, dtype=dtype)

    def run() -> None:
        y = layer(x)
        y.sum().backward()
        layer.zero_grad(set_to_none=True)
        x.grad = None

    x.requires_grad_(True)
    seconds = timed_cuda(run, repeats)
    flops = 6.0 * m * k * n  # forward + dgrad + wgrad approximation
    return {"seconds": seconds, "tflops": flops / seconds / 1e12}


def benchmark_te_fp8(device: torch.device, repeats: int) -> dict[str, Any]:
    try:
        import transformer_engine.pytorch as te
        from transformer_engine.common.recipe import DelayedScaling, Format
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    m, k, n = 8192, 1152, 4608
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16, requires_grad=True)
    layer = te.Linear(k, n, bias=False).to(device=device, dtype=torch.bfloat16)
    recipe = DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=16, amax_compute_algo="max")
    autocast = getattr(te, "autocast", None) or getattr(te, "fp8_autocast")

    def run() -> None:
        try:
            context = autocast(enabled=True, recipe=recipe)
        except TypeError:
            context = autocast(enabled=True, fp8_recipe=recipe)
        with context:
            y = layer(x)
            loss = y.sum()
        loss.backward()
        layer.zero_grad(set_to_none=True)
        x.grad = None

    seconds = timed_cuda(run, repeats)
    flops = 6.0 * m * k * n
    return {"available": True, "seconds": seconds, "tflops": flops / seconds / 1e12}


def benchmark_pcie(device: torch.device, mib: int = 512, repeats: int = 5) -> dict[str, float]:
    count = mib * 2**20 // 2
    host = torch.empty(count, dtype=torch.bfloat16, pin_memory=True)
    gpu = torch.empty_like(host, device=device)
    h2d: list[float] = []
    d2h: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        gpu.copy_(host, non_blocking=True)
        torch.cuda.synchronize(device)
        h2d.append(time.perf_counter() - start)
        start = time.perf_counter()
        host.copy_(gpu, non_blocking=True)
        torch.cuda.synchronize(device)
        d2h.append(time.perf_counter() - start)
    size_gb = mib / 1024
    return {
        "h2d_gbps": size_gb / median(h2d),
        "d2h_gbps": size_gb / median(d2h),
    }


def benchmark_storage(root: Path, mib: int = 512) -> dict[str, float]:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "asterlm-storage-probe.bin"
    block = os.urandom(8 * 2**20)
    started = time.perf_counter()
    with path.open("wb", buffering=0) as handle:
        for _ in range(max(1, mib // 8)):
            handle.write(block)
        os.fsync(handle.fileno())
    write_s = time.perf_counter() - started
    started = time.perf_counter()
    total = 0
    with path.open("rb", buffering=0) as handle:
        while chunk := handle.read(8 * 2**20):
            total += len(chunk)
    read_s = time.perf_counter() - started
    path.unlink(missing_ok=True)
    gib = total / 2**30
    return {"write_gib_s": gib / write_s, "read_gib_s": gib / read_s}


def main() -> None:
    parser = argparse.ArgumentParser(description="Characterize the actual RTX laptop before model selection")
    parser.add_argument("--output", default="runs/hardware-probe.json")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--skip-storage", action="store_true")
    parser.add_argument("--storage-dir", default=None)
    args = parser.parse_args()

    result: dict[str, Any] = {
        "time_unix": time.time(),
        "imports": {
            name: bool(importlib.util.find_spec(name))
            for name in ["fla", "transformer_engine", "torchao", "apollo_torch", "triton"]
        },
    }
    if not torch.cuda.is_available():
        result["status"] = "no_cuda"
    else:
        device = torch.device("cuda")
        torch.backends.cuda.matmul.allow_tf32 = True
        result["status"] = "ok"
        result["system"] = static_system_manifest(device)
        result["telemetry_before"] = SystemSampler(device, 0.1).sample(force=True)
        torch.cuda.empty_cache()
        result["bf16_linear"] = benchmark_linear(device, torch.bfloat16, args.repeats)
        result["fp8_te_linear"] = benchmark_te_fp8(device, args.repeats)
        result["pcie"] = benchmark_pcie(device)
        result["telemetry_after"] = SystemSampler(device, 0.1).sample(force=True)
        if not args.skip_storage:
            storage_root = Path(args.storage_dir) if args.storage_dir else Path(tempfile.gettempdir())
            result["storage"] = benchmark_storage(storage_root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
