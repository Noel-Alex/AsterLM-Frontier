#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from asterlm.artifacts import atomic_write_json
from asterlm.kernels import sparse_gather_attention


def build_causal_sparse_indices(
    sequence: int,
    *,
    batch: int,
    local_window: int,
    compress_ratio: int,
    compressed_topk: int,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """Build deterministic local plus most-recent compressed causal indices."""

    if min(sequence, batch, local_window, compress_ratio, compressed_topk) <= 0:
        raise ValueError("sparse index dimensions must be positive")
    positions = torch.arange(sequence, device=device)
    local_offsets = torch.arange(local_window - 1, -1, -1, device=device)
    local = positions[:, None] - local_offsets[None, :]
    local = torch.where(local >= 0, local, local.new_full((), -1))

    compressed_count = sequence // compress_ratio
    selected_compressed = min(compressed_topk, compressed_count)
    if selected_compressed:
        visible = torch.div(positions + 1, compress_ratio, rounding_mode="floor")
        offsets = torch.arange(selected_compressed, device=device)
        start = (visible - selected_compressed).clamp_min(0)
        compressed = start[:, None] + offsets[None, :]
        valid = offsets[None, :] < visible[:, None].clamp_max(selected_compressed)
        compressed = torch.where(
            valid, compressed + sequence, compressed.new_full((), -1)
        )
        indices = torch.cat((local, compressed), dim=-1)
    else:
        indices = local
    return indices.unsqueeze(0).expand(batch, -1, -1).contiguous(), compressed_count


def _git_state(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(args, cwd=root, text=True).strip()

    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "branch", "--show-current"),
        "dirty": bool(run("git", "status", "--porcelain")),
    }


def _measure(
    function,
    *,
    warmup: int,
    iterations: int,
) -> tuple[float, float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    torch.cuda.synchronize()
    milliseconds = start.elapsed_time(end) / iterations
    return milliseconds, torch.cuda.max_memory_allocated() / 2**30


def _case(
    *,
    sequence: int,
    batch: int,
    heads: int,
    head_dim: int,
    local_window: int,
    compress_ratio: int,
    compressed_topk: int,
    backend: str,
    warmup: int,
    iterations: int,
    reference_memory_limit_gib: float,
) -> dict[str, Any]:
    device = torch.device("cuda")
    indices, compressed_count = build_causal_sparse_indices(
        sequence,
        batch=batch,
        local_window=local_window,
        compress_ratio=compress_ratio,
        compressed_topk=compressed_topk,
        device=device,
    )
    kv_count = sequence + compressed_count
    selected_count = indices.shape[-1]
    # PyTorch materializes selected KV and per-head scores/probabilities. Skip it
    # before OOM rather than turning a reference limitation into a failed result.
    estimated_reference_bytes = (
        batch * sequence * selected_count * head_dim * 2
        + 2 * batch * sequence * heads * selected_count * 4
    )
    if backend == "torch" and estimated_reference_bytes / 2**30 > reference_memory_limit_gib:
        return {
            "sequence": sequence,
            "backend": backend,
            "status": "skipped_reference_memory_guard",
            "estimated_reference_gib": estimated_reference_bytes / 2**30,
        }

    query = torch.randn(batch, sequence, heads, head_dim, device=device, dtype=torch.bfloat16)
    kv = torch.randn(batch, kv_count, head_dim, device=device, dtype=torch.bfloat16)
    sink = torch.zeros(heads, device=device, dtype=torch.float32)

    def inference() -> None:
        with torch.no_grad():
            sparse_gather_attention(query, kv, indices, sink, backend=backend)

    forward_ms, forward_peak_gib = _measure(
        inference, warmup=warmup, iterations=iterations
    )

    train_query = query.detach().requires_grad_(True)
    train_kv = kv.detach().requires_grad_(True)
    train_sink = sink.detach().requires_grad_(True)

    def train_step() -> None:
        for tensor in (train_query, train_kv, train_sink):
            tensor.grad = None
        output = sparse_gather_attention(
            train_query, train_kv, indices, train_sink, backend=backend
        )
        output.float().square().mean().backward()

    train_ms, train_peak_gib = _measure(
        train_step,
        warmup=max(1, warmup // 2),
        iterations=max(1, iterations // 2),
    )
    return {
        "sequence": sequence,
        "backend": backend,
        "status": "ok",
        "batch": batch,
        "heads": heads,
        "head_dim": head_dim,
        "kv_count": kv_count,
        "selected_count": selected_count,
        "forward_ms": forward_ms,
        "forward_tokens_per_second": batch * sequence / (forward_ms / 1000),
        "forward_peak_allocated_gib": forward_peak_gib,
        "forward_backward_ms": train_ms,
        "train_tokens_per_second": batch * sequence / (train_ms / 1000),
        "train_peak_allocated_gib": train_peak_gib,
        "estimated_reference_gib": estimated_reference_bytes / 2**30,
    }


def _parity() -> dict[str, Any]:
    torch.manual_seed(43)
    device = torch.device("cuda")
    query = torch.randn(1, 32, 3, 16, device=device, dtype=torch.float32, requires_grad=True)
    kv = torch.randn(1, 40, 16, device=device, dtype=torch.float32, requires_grad=True)
    indices = torch.randint(0, 40, (1, 32, 9), device=device)
    indices[:, :3, 4:] = -1
    sink = torch.randn(3, device=device, dtype=torch.float32, requires_grad=True)
    optimized_inputs = tuple(
        tensor.detach().clone().requires_grad_(tensor.requires_grad)
        if tensor.is_floating_point()
        else tensor.clone()
        for tensor in (query, kv, indices, sink)
    )
    reference = sparse_gather_attention(query, kv, indices, sink, backend="torch")
    optimized = sparse_gather_attention(*optimized_inputs, backend="triton")
    forward_max_abs = (optimized - reference).abs().max().item()
    gradient = torch.randn_like(reference)
    reference.backward(gradient)
    optimized.backward(gradient)
    gradient_max_abs = {}
    for name, reference_tensor, optimized_tensor in zip(
        ("query", "kv", "sink"),
        (query, kv, sink),
        (optimized_inputs[0], optimized_inputs[1], optimized_inputs[3]),
        strict=True,
    ):
        assert reference_tensor.grad is not None and optimized_tensor.grad is not None
        gradient_max_abs[name] = (
            optimized_tensor.grad - reference_tensor.grad
        ).abs().max().item()
    return {
        "forward_max_abs": forward_max_abs,
        "gradient_max_abs": gradient_max_abs,
        "passed": forward_max_abs <= 2e-4
        and all(value <= 8e-4 for value in gradient_max_abs.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Parity and crossover campaign for Ada sparse attention")
    parser.add_argument("--sequence", type=int, action="append", default=[])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--local-window", type=int, default=128)
    parser.add_argument("--compress-ratio", type=int, default=4)
    parser.add_argument("--compressed-topk", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--reference-memory-limit-gib", type=float, default=4.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/kernel-campaign/sparse-gather-attention/results.json"),
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the sparse-attention kernel campaign")
    root = Path(__file__).resolve().parents[1]
    sequences = args.sequence or [512, 2048, 8192, 32768]
    started = time.time()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at_unix": started,
        "source": _git_state(root),
        "system": {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
        },
        "arguments": vars(args) | {"output": args.output.as_posix()},
        "parity": None,
        "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, payload)
    try:
        payload["parity"] = _parity()
        if not payload["parity"]["passed"]:
            raise RuntimeError(f"Triton parity failed: {json.dumps(payload['parity'])}")
        for sequence in sequences:
            for backend in ("torch", "triton"):
                result = _case(
                    sequence=sequence,
                    batch=args.batch,
                    heads=args.heads,
                    head_dim=args.head_dim,
                    local_window=args.local_window,
                    compress_ratio=args.compress_ratio,
                    compressed_topk=args.compressed_topk,
                    backend=backend,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    reference_memory_limit_gib=args.reference_memory_limit_gib,
                )
                payload["results"].append(result)
                atomic_write_json(args.output, payload)
        payload["status"] = "complete"
    except BaseException as exc:
        payload["status"] = "failed"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        payload["finished_at_unix"] = time.time()
        payload["duration_seconds"] = payload["finished_at_unix"] - started
        atomic_write_json(args.output, payload)


if __name__ == "__main__":
    main()

