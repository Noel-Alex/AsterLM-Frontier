#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from asterlm.cache import LatentLayerCache


def benchmark(scheme: str, latent: torch.Tensor, rope: torch.Tensor, args) -> dict:
    cache = LatentLayerCache(
        cache_dtype=scheme,
        group_size=args.group_size,
        recent_tokens=args.recent_tokens,
        chunk_tokens=args.chunk_tokens,
        quantize_rope=args.quantize_rope,
    )
    started = time.perf_counter()
    for start in range(0, latent.shape[1], args.append_tokens):
        cache.append(
            latent[:, start : start + args.append_tokens],
            rope[:, start : start + args.append_tokens],
            window_size=None,
            sink_tokens=0,
        )
    if latent.is_cuda:
        torch.cuda.synchronize()
    encode_s = time.perf_counter() - started
    started = time.perf_counter()
    restored_latent = cache.materialize_latent(dtype=latent.dtype, device=latent.device)
    restored_rope = cache.materialize_rope(dtype=rope.dtype, device=rope.device)
    if latent.is_cuda:
        torch.cuda.synchronize()
    decode_s = time.perf_counter() - started
    assert restored_latent is not None and restored_rope is not None
    latent_error = (restored_latent - latent).float()
    rope_error = (restored_rope - rope).float()
    reference_bytes = (latent.numel() + rope.numel()) * 2
    return {
        "scheme": scheme,
        "tokens": latent.shape[1],
        "cache_mib": cache.num_bytes / 2**20,
        "bf16_reference_mib": reference_bytes / 2**20,
        "compression_ratio": reference_bytes / max(cache.num_bytes, 1),
        "latent_rmse": float(latent_error.square().mean().sqrt()),
        "latent_max_abs": float(latent_error.abs().max()),
        "rope_rmse": float(rope_error.square().mean().sqrt()),
        "encode_seconds": encode_s,
        "materialize_seconds": decode_s,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare latent-KV cache storage schemes")
    parser.add_argument("--tokens", type=int, default=131072)
    parser.add_argument("--latent-rank", type=int, default=96)
    parser.add_argument("--rope-dim", type=int, default=32)
    parser.add_argument("--recent-tokens", type=int, default=1024)
    parser.add_argument("--chunk-tokens", type=int, default=2048)
    parser.add_argument("--append-tokens", type=int, default=2048)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--quantize-rope", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="runs/cache-quantization.json")
    args = parser.parse_args()
    device = torch.device(args.device)
    generator = torch.Generator(device=device).manual_seed(1337)
    # Scale approximates post-RMSNorm latent values; measure on real checkpoints later.
    latent = torch.randn(1, args.tokens, args.latent_rank, device=device, dtype=torch.bfloat16, generator=generator)
    rope = torch.randn(1, args.tokens, args.rope_dim, device=device, dtype=torch.bfloat16, generator=generator)
    results = [
        benchmark(scheme, latent, rope, args)
        for scheme in ("bfloat16", "float8", "int8", "int4", "hadamard_int4")
        if scheme != "float8" or hasattr(torch, "float8_e4m3fn")
    ]
    payload = {"arguments": vars(args), "results": results}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
