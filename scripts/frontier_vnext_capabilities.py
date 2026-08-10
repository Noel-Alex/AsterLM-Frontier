#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch

from asterlm.runtime import configure_transformer_engine_runtime

configure_transformer_engine_runtime()


def version_of(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception:
        return None


def run_check(name: str, fn: Callable[[], Any]) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        detail = fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return {
            "name": name,
            "status": "ok",
            "seconds": time.perf_counter() - started,
            "detail": detail,
        }
    except Exception as exc:
        return {
            "name": name,
            "status": "error",
            "seconds": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=8),
        }


def check_linear_ce() -> dict[str, Any]:
    fn = getattr(torch.nn.functional, "linear_cross_entropy", None)
    options_cls = getattr(torch.nn, "LinearCrossEntropyOptions", None)
    if fn is None or options_cls is None:
        raise RuntimeError("PyTorch LinearCrossEntropy API is unavailable")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    torch.manual_seed(100)
    x = torch.randn(512, 256, device=device, dtype=dtype, requires_grad=True)
    w = torch.randn(4096, 256, device=device, dtype=dtype, requires_grad=True)
    y = torch.randint(0, 4096, (512,), device=device)
    options = options_cls(chunking_method="auto", acc_policy="compact")
    loss = fn(x, w, y, options=options)
    loss.backward()
    return {
        "loss": float(loss.detach()),
        "x_grad_finite": bool(torch.isfinite(x.grad).all()),
        "w_grad_finite": bool(torch.isfinite(w.grad).all()),
    }


def check_flex_attention() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    device = torch.device("cuda")
    q = torch.randn(1, 4, 512, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, 4, 512, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(1, 4, 512, 64, device=device, dtype=torch.bfloat16, requires_grad=True)

    def mask_mod(b, h, qi, ki):
        del b, h
        return (ki <= qi) & ((ki >= qi - 127) | (ki < 16) | ((ki % 128) == 0))

    mask = create_block_mask(mask_mod, 1, 4, 512, 512, device=device, BLOCK_SIZE=128)
    out = flex_attention(q, k, v, block_mask=mask)
    loss = out.float().square().mean()
    loss.backward()
    return {
        "loss": float(loss.detach()),
        "sparsity_percent": float(mask.sparsity()),
        "grad_finite": all(bool(torch.isfinite(t.grad).all()) for t in (q, k, v)),
    }


def check_fla() -> dict[str, Any]:
    from fla.layers.gdn2 import GatedDeltaNet2
    from fla.layers.kda import KimiDeltaAttention
    from fla.ops.attnres import fused_attnres

    return {
        "kda": f"{KimiDeltaAttention.__module__}.{KimiDeltaAttention.__name__}",
        "gdn2": f"{GatedDeltaNet2.__module__}.{GatedDeltaNet2.__name__}",
        "fused_attnres": callable(fused_attnres),
    }


def _te_recipe_smoke(recipe_name: str) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import DelayedScaling, Float8CurrentScaling, Format

    recipe = (
        Float8CurrentScaling(fp8_format=Format.HYBRID)
        if recipe_name == "current"
        else DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=16, amax_compute_algo="max")
    )
    layer = te.Linear(256, 256, bias=False, params_dtype=torch.bfloat16).cuda()
    x = torch.randn(16, 64, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    with te.autocast(enabled=True, recipe=recipe):
        out = layer(x)
        loss = out.float().square().mean()
    loss.backward()
    return {
        "loss": float(loss.detach()),
        "x_grad_finite": bool(torch.isfinite(x.grad).all()),
        "weight_grad_finite": bool(torch.isfinite(layer.weight.grad).all()),
    }


def check_attnres_cuda() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    from asterlm.layers.attnres_vnext import AttnResMix

    layer = AttnResMix(128).cuda().to(torch.bfloat16)
    residuals = [
        torch.randn(1, 256, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for _ in range(4)
    ]
    out = layer(residuals)
    loss = out.float().square().mean()
    loss.backward()
    return {
        "loss": float(loss.detach()),
        "fused_available": layer.fused_available(),
        "grad_finite": all(bool(torch.isfinite(x.grad).all()) for x in residuals),
    }


def check_grouped_moe() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import DelayedScaling, Format
    from asterlm.layers.moe import DeepSeekStyleMoE

    old = os.environ.get("ASTER_MOE_IMPL")
    os.environ["ASTER_MOE_IMPL"] = "grouped"
    try:
        layer = DeepSeekStyleMoE(
            128,
            256,
            8,
            2,
            1,
            0.0,
            "sigmoid",
            "hybrid",
            0.001,
            "transformer_engine",
        ).cuda().to(torch.bfloat16)
        x = torch.randn(1, 256, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        recipe = DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=16, amax_compute_algo="max")
        with te.autocast(enabled=True, recipe=recipe):
            y = layer(x)
            loss = y.float().square().mean()
        loss.backward()
        return {
            "loss": float(loss.detach()),
            "grad_finite": bool(torch.isfinite(x.grad).all()),
            "expert_load": None if layer.last_load is None else [float(v) for v in layer.last_load.cpu()],
        }
    finally:
        if old is None:
            os.environ.pop("ASTER_MOE_IMPL", None)
        else:
            os.environ["ASTER_MOE_IMPL"] = old


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/frontier-vnext/capabilities.json")
    args = parser.parse_args()

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu = {
            "name": props.name,
            "total_memory_gib": props.total_memory / 2**30,
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "multiprocessors": props.multi_processor_count,
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        }
    else:
        gpu = {"available": False}

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": gpu,
        "versions": {
            "transformer_engine": version_of("transformer_engine"),
            "fla": version_of("fla"),
            "triton": version_of("triton"),
            "torchao": version_of("torchao"),
            "apollo_torch": version_of("apollo_torch"),
        },
        "checks": [],
    }

    checks = [
        ("pytorch_linear_cross_entropy", check_linear_ce),
        ("fla_frontier_imports", check_fla),
        ("flex_attention_cuda", check_flex_attention),
        ("te_fp8_delayed", lambda: _te_recipe_smoke("delayed")),
        ("te_fp8_current", lambda: _te_recipe_smoke("current")),
        ("fla_attnres_cuda", check_attnres_cuda),
        ("te_grouped_moe_cuda", check_grouped_moe),
    ]
    for name, fn in checks:
        print(f"[capability] {name}", flush=True)
        result = run_check(name, fn)
        report["checks"].append(result)
        print(f"  -> {result['status']} ({result['seconds']:.2f}s)", flush=True)
        if result["status"] != "ok":
            print(f"     {result.get('error')}", flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"Capability report: {output}")


if __name__ == "__main__":
    main()
