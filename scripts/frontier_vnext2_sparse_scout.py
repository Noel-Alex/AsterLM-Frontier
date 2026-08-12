#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import subprocess
import sys
import time
import types
from pathlib import Path

import torch
import yaml

ROOT = Path.cwd().resolve()
PIN = "bd67af59b90afa34b25f61d2922e612d10dba3bd"
REPO = ROOT / ".cache/third_party/native-sparse-attention"


def ensure_repo() -> tuple[bool, str]:
    try:
        if not (REPO / ".git").is_dir():
            REPO.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "clone", "--filter=blob:none", "https://github.com/fla-org/native-sparse-attention.git", str(REPO)],
                cwd=ROOT, check=True, timeout=300,
            )
        subprocess.run(["git", "fetch", "origin", PIN], cwd=REPO, check=False, timeout=180)
        subprocess.run(["git", "checkout", "--detach", PIN], cwd=REPO, check=True, timeout=120)
        return True, PIN
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def bench_sdpa(seq: int, q_heads: int, qk_dim: int, value_dim: int, repeats: int) -> dict:
    cleanup()
    device = "cuda"
    q = torch.randn(1, q_heads, seq, qk_dim, device=device, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, 1, seq, qk_dim, device=device, dtype=torch.bfloat16, requires_grad=True)
    v0 = torch.randn(1, 1, seq, value_dim, device=device, dtype=torch.bfloat16, requires_grad=True)
    v = torch.nn.functional.pad(v0, (0, qk_dim - value_dim))
    scale = 1 / math.sqrt(96)  # Aster frontier original MLA head_dim(64)+RoPE(32) scale
    durations = []
    for i in range(repeats + 1):
        for t in (q, k, v0):
            t.grad = None
        torch.cuda.synchronize()
        started = time.perf_counter()
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True, enable_gqa=True, scale=scale
        )[..., :value_dim]
        loss = out.float().square().mean()
        loss.backward()
        torch.cuda.synchronize()
        dt = time.perf_counter() - started
        if i:
            durations.append(dt)
    return {
        "status": "ok",
        "seconds_median": statistics.median(durations),
        "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
        "loss": float(loss.detach()),
        "finite": bool(torch.isfinite(q.grad).all() and torch.isfinite(k.grad).all() and torch.isfinite(v0.grad).all()),
    }


def load_parallel_nsa():
    """Load the pinned NSA operator against Aster's installed FLA runtime."""

    # The pinned NSA source imports FLA 0.4's former `fla.ops.common.utils`
    # location. FLA 0.5 exposes the same helpers from `fla.ops.utils`. Alias the
    # module in memory so the isolated scout uses Aster's tested FLA runtime,
    # without installing NSA's historical FLA submodule over the environment.
    from fla.ops import utils as fla_ops_utils

    fla_common = types.ModuleType("fla.ops.common")
    fla_common.__path__ = []
    fla_common.utils = fla_ops_utils
    sys.modules["fla.ops.common"] = fla_common
    sys.modules["fla.ops.common.utils"] = fla_ops_utils

    # Load only the pinned kernel package. Adding REPO to sys.path would expose
    # its broken/uninitialized `fla` symlink ahead of the installed FLA runtime.
    nsa_package = types.ModuleType("native_sparse_attention")
    nsa_package.__path__ = [str(REPO / "native_sparse_attention")]
    sys.modules["native_sparse_attention"] = nsa_package
    from native_sparse_attention.ops import parallel_nsa

    return parallel_nsa


def bench_nsa(seq: int, q_heads: int, qk_dim: int, value_dim: int, repeats: int) -> dict:
    cleanup()
    parallel_nsa = load_parallel_nsa()
    q = torch.randn(1, seq, q_heads, qk_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, seq, 1, qk_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(1, seq, 1, value_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    # Selected + compressed branches only. The official sliding branch depends on
    # flash-attn; this scout deliberately avoids mutating Aster's environment just
    # to install it. Gates are fixed because we are measuring kernel/system behavior.
    g_cmp = torch.full((1, seq, q_heads), 0.5, device="cuda", dtype=torch.bfloat16)
    g_slc = torch.ones((1, seq, q_heads), device="cuda", dtype=torch.bfloat16)
    g_swa = torch.zeros((1, seq, q_heads), device="cuda", dtype=torch.bfloat16)
    durations = []
    for i in range(repeats + 1):
        for t in (q, k, v):
            t.grad = None
        torch.cuda.synchronize()
        started = time.perf_counter()
        out = parallel_nsa(
            q=q,
            k=k,
            v=v,
            g_cmp=g_cmp,
            g_slc=g_slc,
            g_swa=g_swa,
            block_counts=16,
            block_size=64,
            window_size=0,
            scale=1 / math.sqrt(96),  # preserve Aster MLA softmax scale after absorption
        )
        loss = out.float().square().mean()
        loss.backward()
        torch.cuda.synchronize()
        dt = time.perf_counter() - started
        if i:
            durations.append(dt)
    return {
        "status": "ok",
        "seconds_median": statistics.median(durations),
        "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
        "loss": float(loss.detach()),
        "finite": bool(torch.isfinite(q.grad).all() and torch.isfinite(k.grad).all() and torch.isfinite(v.grad).all()),
        "branches": "compression+selected; sliding omitted because flash-attn is not installed by this scout",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/frontier-vnext2/results/sparse-scout.json")
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "started",
        "pin": PIN,
        "aster_compatibility": {},
        "benchmarks": [],
    }
    try:
        base_cfg = yaml.safe_load((ROOT / "configs/model/aster_moe_frontier_893m_fp8.yaml").read_text())["model"]
        aster_heads = int(base_cfg["n_heads"])
        result["aster_compatibility"] = {
            "query_heads": aster_heads,
            "native_nsa_selected_kernel_requires": "HQ/H must be a multiple of 16; with one absorbed KV head this means HQ % 16 == 0",
            "directly_compatible": aster_heads % 16 == 0,
            "decision": (
                "direct scout is architecture-compatible" if aster_heads % 16 == 0 else
                "do not integrate native NSA into Aster yet; benchmark a 16-head surrogate only, then decide whether an arbitrary-group (G=18) Ada kernel or a head-layout redesign is justified"
            ),
        }
        ok, detail = ensure_repo()
        result["checkout"] = {"ok": ok, "detail": detail}
        if not ok:
            result["status"] = "skipped"
            return
        # Aster frontier absorbed Q/K width is latent_rank + rope_dim = 128; value width is latent_rank = 96.
        # Native NSA requires a >=16 GQA group, so use HQ=16 as a kernel-potential surrogate.
        for seq in (8192, 16384):
            row = {"sequence": seq, "q_heads": 16, "kv_heads": 1, "qk_dim": 128, "value_dim": 96, "softmax_scale_dim": 96}
            try:
                row["sdpa"] = bench_sdpa(seq, 16, 128, 96, args.repeats)
            except Exception as exc:
                row["sdpa"] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
            cleanup()
            try:
                row["nsa"] = bench_nsa(seq, 16, 128, 96, args.repeats)
            except torch.cuda.OutOfMemoryError as exc:
                row["nsa"] = {"status": "oom", "error": str(exc)}
            except Exception as exc:
                row["nsa"] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
            cleanup()
            result["benchmarks"].append(row)
        result["status"] = "ok"
    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
