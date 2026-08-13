#!/usr/bin/env python3
"""Certify cached laptop inference for the exact frozen 1.448B architecture."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
GATE_ID = "laptop_inference"
EXPECTED_TOTAL_PARAMETERS = 1_448_120_880
EXPECTED_ACTIVE_PARAMETERS = 568_155_376
EXPECTED_LAYERS = 32


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2) + "\n")


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_clean() -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError("Laptop inference evidence requires a clean checkout")


def _ensure_allocator_reexec() -> None:
    environment = dict(os.environ)
    modern = environment.get("PYTORCH_ALLOC_CONF")
    legacy = environment.get("PYTORCH_CUDA_ALLOC_CONF")
    if modern and legacy and modern != legacy:
        raise RuntimeError("PYTORCH_ALLOC_CONF and PYTORCH_CUDA_ALLOC_CONF disagree")
    selected = modern or legacy or "expandable_segments:True"
    environment["PYTORCH_ALLOC_CONF"] = selected
    environment["PYTORCH_CUDA_ALLOC_CONF"] = selected
    if os.environ.get("ASTER_INFERENCE_CANARY_REEXEC") == "1":
        return
    if any(
        os.environ.get(key) != environment[key]
        for key in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")
    ):
        environment["ASTER_INFERENCE_CANARY_REEXEC"] = "1"
        result = subprocess.run(
            [sys.executable, *sys.argv], cwd=ROOT, env=environment, check=False
        )
        raise SystemExit(result.returncode)


def _active_compute_processes() -> list[dict[str, Any]]:
    output = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
            "-i",
            "0",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout
    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        values = [value.strip() for value in line.split(",")]
        if values and values[0]:
            rows.append(
                {
                    "pid": int(values[0]),
                    "process_name": values[1] if len(values) > 1 else "unknown",
                    "used_gpu_memory_mib": values[2] if len(values) > 2 else "unknown",
                }
            )
    return rows


def validate_measurement(result: dict[str, Any]) -> None:
    architecture = result.get("architecture") or {}
    runtime = result.get("runtime") or {}
    if result.get("status") != "passed":
        raise ValueError("Inference canary did not pass")
    if architecture.get("effective_parameters") != EXPECTED_TOTAL_PARAMETERS:
        raise ValueError("Inference canary used the wrong total parameter count")
    if architecture.get("active_parameters_estimate") != EXPECTED_ACTIVE_PARAMETERS:
        raise ValueError("Inference canary used the wrong active parameter count")
    if architecture.get("n_layers") != EXPECTED_LAYERS:
        raise ValueError("Inference canary used the wrong layer count")
    if runtime.get("executed_block_count") != EXPECTED_LAYERS:
        raise ValueError("Inference canary did not execute every model block")
    if runtime.get("decoded_tokens", 0) < 2:
        raise ValueError("Inference canary did not exercise repeated cached decode")
    if runtime.get("cache_bytes", 0) <= 0:
        raise ValueError("Inference canary did not populate its cache")
    for key in ("prefill_tokens_per_second", "decode_tokens_per_second", "peak_allocated_gib"):
        value = float(runtime.get(key, math.nan))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"Inference canary has invalid {key}")
    if runtime.get("finite_logits") is not True:
        raise ValueError("Inference canary produced non-finite logits")


def _update_ledger(path: Path, proof: dict[str, str]) -> None:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for gate in payload.get("gates", []):
        if gate.get("id") == GATE_ID:
            gate["status"] = "passed"
            gate["evidence"] = [proof]
            gate.pop("note", None)
            _atomic_write_text(path, yaml.safe_dump(payload, sort_keys=False))
            return
    raise RuntimeError(f"Promotion ledger is missing {GATE_ID}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="configs/model/aster_k3_latentmoe_1p45b_a568m.yaml")
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--decode-tokens", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--gates", type=Path, default=ROOT / "configs/experiments/promotion_gates.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/promotion-evidence")
    args = parser.parse_args()
    _require_clean()
    _ensure_allocator_reexec()

    import torch

    from asterlm import AsterConfig, AsterLM
    from asterlm.training.telemetry import static_system_manifest

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected laptop CUDA BF16 runtime is unavailable")
    device = torch.device("cuda:0")
    hardware = static_system_manifest(device)
    gpu_name = torch.cuda.get_device_name(device)
    if "RTX 4080 Laptop" not in gpu_name:
        raise RuntimeError(f"Expected the selected RTX 4080 Laptop GPU, got {gpu_name!r}")
    active_before = _active_compute_processes()
    if active_before:
        raise RuntimeError(f"Inference canary requires an exclusive GPU: {active_before}")

    commit = _git_commit()
    model_path = (ROOT / args.model).resolve()
    config = AsterConfig.from_yaml(model_path)
    config.gradient_checkpointing = False
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    setup_started = time.perf_counter()
    model = AsterLM(config, named_initialization_seed=args.seed, moe_implementation="cutlass")
    model = model.to(device=device, dtype=torch.bfloat16)
    for name, parameter in model.named_parameters():
        if name.endswith(("A_log", "dt_bias")):
            parameter.data = parameter.data.float()
    packed = model.pack_grouped_expert_storage()
    model.eval()
    setup_seconds = time.perf_counter() - setup_started

    effective_parameters = model.effective_parameter_count()
    active_parameters = model.active_parameter_count()
    hooks = []
    executed: set[int] = set()
    for index, block in enumerate(model.blocks):
        hooks.append(block.register_forward_hook(lambda _m, _i, _o, index=index: executed.add(index)))

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 1)
    prompt = torch.randint(
        7,
        config.vocab_size,
        (1, args.prompt_tokens),
        generator=generator,
        device=device,
    )
    cache = model.make_cache()
    with torch.inference_mode():
        torch.cuda.synchronize(device)
        prefill_started = time.perf_counter()
        output = model(prompt, cache=cache, use_cache=True)
        torch.cuda.synchronize(device)
        prefill_seconds = time.perf_counter() - prefill_started
        finite_logits = bool(torch.isfinite(output.logits).all().item())
        token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        decode_durations: list[float] = []
        for _ in range(args.decode_tokens):
            torch.cuda.synchronize(device)
            decode_started = time.perf_counter()
            output = model(token, cache=cache, use_cache=True)
            torch.cuda.synchronize(device)
            decode_durations.append(time.perf_counter() - decode_started)
            finite_logits = finite_logits and bool(torch.isfinite(output.logits).all().item())
            token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
    for hook in hooks:
        hook.remove()

    result = {
        "schema_version": 1,
        "status": "passed",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": commit,
        "model": {"path": model_path.relative_to(ROOT).as_posix(), "sha256": _sha256_file(model_path)},
        "hardware": hardware,
        "architecture": {
            **model.architecture_summary(),
            "effective_parameters": effective_parameters,
            "active_parameters_estimate": active_parameters,
            "n_layers": len(model.blocks),
            "moe_implementation": model.moe_implementation,
            "packed_grouped_expert_storage": packed,
        },
        "runtime": {
            "dtype": "bfloat16",
            "prompt_tokens": args.prompt_tokens,
            "decoded_tokens": args.decode_tokens,
            "executed_block_count": len(executed),
            "executed_block_indices": sorted(executed),
            "finite_logits": finite_logits,
            "setup_seconds": setup_seconds,
            "prefill_seconds": prefill_seconds,
            "prefill_tokens_per_second": args.prompt_tokens / prefill_seconds,
            "decode_seconds": sum(decode_durations),
            "decode_tokens_per_second": args.decode_tokens / sum(decode_durations),
            "decode_latency_median_ms": statistics.median(decode_durations) * 1000.0,
            "cache_bytes": cache.num_bytes,
            "cache_mib": cache.num_bytes / 2**20,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        },
        "claims": {
            "certifies": "selected architecture cached BF16 prefill/decode runtime on the laptop",
            "does_not_certify": [
                "language quality from random initialization",
                "long-context retrieval quality",
                "256K or 1M latency",
                "post-training behavior",
            ],
        },
    }
    validate_measurement(result)
    output = args.output / commit[:12] / "laptop-inference"
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "laptop-inference-result.json"
    _atomic_write_json(result_path, result)
    artifact = {"path": result_path.relative_to(ROOT).as_posix(), "sha256": _sha256_file(result_path)}
    proof_path = output / f"{GATE_ID}-proof.json"
    _atomic_write_json(
        proof_path,
        {
            "schema_version": 1,
            "gate_id": GATE_ID,
            "status": "passed",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "evaluator": {"name": "asterlm-laptop-inference-canary", "version": "1", "git_commit": commit},
            "experiment_ids": [f"laptop-inference-{commit[:12]}"],
            "artifacts": [artifact],
        },
    )
    _update_ledger(
        args.gates,
        {"path": proof_path.relative_to(ROOT).as_posix(), "sha256": _sha256_file(proof_path)},
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
