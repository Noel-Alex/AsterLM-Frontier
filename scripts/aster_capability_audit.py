#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def symbol_available(module: str, symbol: str) -> bool:
    try:
        loaded = __import__(module, fromlist=[symbol])
        getattr(loaded, symbol)
        return True
    except Exception:
        return False


def contains(path: str, token: str) -> bool:
    target = ROOT / path
    if not target.is_file():
        return False
    try:
        return token in target.read_text(encoding="utf-8")
    except Exception:
        return False


def nvidia_info() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,memory.used,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
        if not out:
            return {"available": False}
        first = [item.strip() for item in out.splitlines()[0].split(",")]
        return {
            "available": True,
            "name": first[0] if len(first) > 0 else None,
            "driver": first[1] if len(first) > 1 else None,
            "memory_total_mib": float(first[2]) if len(first) > 2 else None,
            "memory_used_mib": float(first[3]) if len(first) > 3 else None,
            "compute_capability": first[4] if len(first) > 4 else None,
        }
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def check(
    ident: str,
    label: str,
    status: str,
    detail: str,
    *,
    runtime: bool | None = None,
    evidence: list[str] | None = None,
    research: str | None = None,
) -> dict[str, Any]:
    return {
        "id": ident,
        "label": label,
        "status": status,
        "runtime_available": runtime,
        "detail": detail,
        "evidence": evidence or [],
        "research": research,
    }


def build_report() -> dict[str, Any]:
    # Aster needs the layer package, not merely fla-core's low-level kernels.
    fla = symbol_available("fla.layers.kda", "KimiDeltaAttention")
    torchao = module_available("torchao")
    te = module_available("transformer_engine")
    apollo = module_available("apollo_torch") or module_available("apollo")
    wandb = module_available("wandb")
    tensorboard = module_available("tensorboard")

    checks = [
        check(
            "kda",
            "Kimi Delta Attention",
            "implemented",
            "Aster wraps fla.layers.kda.KimiDeltaAttention. The pure-PyTorch fallback is explicitly correctness-only.",
            runtime=fla,
            evidence=["src/asterlm/layers/kda.py"],
            research="Kimi Linear (2025)",
        ),
        check(
            "mla",
            "MLA-style latent attention",
            "implemented",
            "Compressed latent KV + decoupled RoPE channel, PyTorch SDPA for full attention and absorbed online-softmax single-token decode.",
            runtime=contains("src/asterlm/layers/latent_attention.py", "_absorbed_decode"),
            evidence=["src/asterlm/layers/latent_attention.py"],
            research="DeepSeek MLA / Kimi Linear",
        ),
        check(
            "kv_int4",
            "Hadamard INT4 hot/cold latent cache",
            "implemented",
            "The newest cache remains in compute precision while older chunks can be quantized; the cache tracks actual byte usage.",
            runtime=contains("src/asterlm/cache.py", "hadamard") or contains("src/asterlm/quantization/kv.py", "hadamard"),
            evidence=["src/asterlm/cache.py", "src/asterlm/quantization/kv.py"],
        ),
        check(
            "mtp_train",
            "Multi-token prediction training",
            "implemented",
            "MTP heads and auxiliary loss are part of AsterLM training.",
            runtime=contains("src/asterlm/model.py", "MultiTokenPredictor"),
            evidence=["src/asterlm/model.py", "src/asterlm/layers/mtp.py"],
            research="DeepSeek-V3 MTP",
        ),
        check(
            "mtp_decode",
            "MTP self-speculative decoding",
            "reference",
            "The current generate_mtp_greedy verifier is exact but performs full-prefix verification. It is a quality/acceptance reference, not a production speed path.",
            runtime=contains("src/asterlm/generation/speculative.py", "generate_mtp_greedy"),
            evidence=["src/asterlm/generation/speculative.py"],
        ),
        check(
            "deepspec",
            "DeepSpec / DSpark / DFlash / Eagle3",
            "not-integrated",
            "Do not label Aster as DeepSpec-enabled yet. DeepSeek's released DeepSpec stack currently ships Qwen/Gemma targets and Aster has no DSpark/DFlash adapter.",
            runtime=False,
            research="DeepSpec / DSpark (2026)",
        ),
        check(
            "gdn2",
            "Gated DeltaNet-2",
            "research-gap",
            "Newer than Aster's KDA path. It decouples channel-wise erase/write gates. Not integrated into the stable Aster architecture.",
            runtime=False,
            research="Gated DeltaNet-2 (May 2026)",
        ),
        check(
            "lca",
            "Latent-Condensed Attention",
            "research-gap",
            "ACL 2026 MLA-native context condensation is not implemented in Aster.",
            runtime=False,
            research="Latent-Condensed Transformer (ACL 2026)",
        ),
        check(
            "nha",
            "Native Hybrid Attention",
            "research-gap",
            "ACL 2026 intra/inter-layer hybrid attention is not implemented; Aster uses the Kimi-style KDA/MLA layer ratio instead.",
            runtime=False,
            research="Native Hybrid Attention (ACL 2026)",
        ),
        check(
            "robsa",
            "RoBSA sparse MLA decoding",
            "research-gap",
            "ACL 2026 training-free sparse decoding for MLA is not implemented in Aster's latent-attention decode path.",
            runtime=False,
            research="RoBSA (ACL 2026)",
        ),
        check(
            "moe",
            "DeepSeek-style MoE routing",
            "reference",
            "Top-k routed + shared experts, bias/hybrid balancing and router diagnostics are implemented. Dispatch is a Python expert loop, not fused grouped GEMM.",
            runtime=contains("src/asterlm/layers/moe.py", "DeepSeekStyleMoE"),
            evidence=["src/asterlm/layers/moe.py"],
        ),
        check(
            "yarn",
            "YaRN long-context scaling",
            "implemented",
            "YaRN parameters are represented in the model config/rotary path. Long context still requires empirical retrieval validation.",
            runtime=contains("src/asterlm/layers/rotary.py", "yarn") or contains("src/asterlm/config.py", "yarn"),
            evidence=["src/asterlm/layers/rotary.py", "src/asterlm/config.py"],
        ),
        check(
            "apollo",
            "APOLLO optimizer",
            "optional-runtime",
            "Training configs can use APOLLO Mini; runtime availability depends on the local extra.",
            runtime=apollo,
            evidence=["src/asterlm/optim/hybrid.py"],
        ),
        check(
            "torchao",
            "TorchAO low-bit / CPU-offload optimizers",
            "optional-runtime",
            "Used by the long-context and quantization experiments when installed.",
            runtime=torchao,
            evidence=["src/asterlm/optim/hybrid.py"],
        ),
        check(
            "fp8",
            "Transformer Engine FP8",
            "optional-runtime",
            "Aster has a separate TE/FP8 experiment path. It must be benchmarked independently on the live GPU.",
            runtime=te,
            evidence=["configs/model/aster_moe_frontier_893m_fp8.yaml"],
        ),
        check(
            "telemetry",
            "Training telemetry + diagnostic bundles",
            "implemented",
            "JSONL, TensorBoard, optional W&B, system/GPU telemetry, gradient/parameter/MoE diagnostics and failure bundles are wired into Trainer.",
            runtime=True,
            evidence=["src/asterlm/training/engine.py", "src/asterlm/training/telemetry.py"],
        ),
        check(
            "checkpoint",
            "Atomic resumable checkpoints",
            "implemented",
            "Model/optimizer/RNG/token state, rolling retention and permanent milestones are supported.",
            runtime=contains("src/asterlm/training/engine.py", "milestone_tokens"),
            evidence=["src/asterlm/training/checkpoint.py", "src/asterlm/training/engine.py"],
        ),
        check(
            "tracking_wandb",
            "Weights & Biases",
            "optional-runtime",
            "Training configs request W&B tracking; local install is checked here.",
            runtime=wandb,
        ),
        check(
            "tracking_tensorboard",
            "TensorBoard",
            "optional-runtime",
            "Training configs request TensorBoard; local install is checked here.",
            runtime=tensorboard,
        ),
    ]

    return {
        "version": 1,
        "repo": str(ROOT),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "gpu": nvidia_info(),
        "checks": checks,
        "summary": {
            "implemented": sum(item["status"] == "implemented" for item in checks),
            "reference": sum(item["status"] == "reference" for item in checks),
            "optional_runtime": sum(item["status"] == "optional-runtime" for item in checks),
            "research_gaps": sum(item["status"] in {"research-gap", "not-integrated"} for item in checks),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit AsterLM architecture/runtime capability claims")
    parser.add_argument("--json", default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Also run the focused model/speculation/MoE tests. This can take longer.",
    )
    args = parser.parse_args()

    report = build_report()
    if args.smoke:
        tests = [
            "tests/test_speculative.py",
            "tests/test_moe.py",
        ]
        existing = [item for item in tests if (ROOT / item).is_file()]
        if existing:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", *existing],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            report["smoke"] = {
                "returncode": result.returncode,
                "ok": result.returncode == 0,
                "output": result.stdout[-12000:],
            }
        else:
            report["smoke"] = {"ok": False, "error": "Focused test files were not found."}

    if args.json:
        target = Path(args.json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
