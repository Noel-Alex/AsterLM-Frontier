#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import torch

from asterlm import AsterConfig, AsterLM, TrainConfig
from asterlm.optim import build_optimizer
from asterlm.training.precision import PrecisionManager
from asterlm.training.telemetry import SystemSampler, static_system_manifest


def gib(value: int | float) -> float:
    return float(value) / 2**30


def cuda_snapshot(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}
    stats = torch.cuda.memory_stats(device)
    return {
        "allocated_gib": gib(torch.cuda.memory_allocated(device)),
        "reserved_gib": gib(torch.cuda.memory_reserved(device)),
        "peak_allocated_gib": gib(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_gib": gib(torch.cuda.max_memory_reserved(device)),
        "inactive_split_gib": gib(stats.get("inactive_split_bytes.all.current", 0)),
    }


def parameter_storage(model: torch.nn.Module) -> dict[str, Any]:
    result: dict[str, Any] = {
        "total_gib": 0.0,
        "parameter_gib": 0.0,
        "buffer_gib": 0.0,
        "by_dtype": {},
    }
    seen: set[int] = set()
    parameter_total = 0
    buffer_total = 0

    def account(tensor: torch.Tensor, kind: str) -> None:
        nonlocal parameter_total, buffer_total
        if id(tensor) in seen:
            return
        seen.add(id(tensor))
        size = tensor.numel() * tensor.element_size()
        if kind == "parameter":
            parameter_total += size
        else:
            buffer_total += size
        key = f"{kind}:{str(tensor.dtype).removeprefix('torch.')}"
        result["by_dtype"][key] = result["by_dtype"].get(key, 0) + size

    for tensor in model.parameters():
        account(tensor, "parameter")
    for tensor in model.buffers():
        account(tensor, "buffer")

    result["parameter_gib"] = gib(parameter_total)
    result["buffer_gib"] = gib(buffer_total)
    result["total_gib"] = gib(parameter_total + buffer_total)
    result["by_dtype"] = {key: gib(value) for key, value in result["by_dtype"].items()}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure actual AsterLM VRAM before corpus training")
    parser.add_argument("--model", default="configs/model/aster_moe_frontier_893m_a484m.yaml")
    parser.add_argument("--train-config", default="configs/train/probe_memory_matrix.yaml")
    parser.add_argument("--sequence", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--accum", type=int, default=None)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--optimizer",
        choices=[
            "muon_adamw",
            "apollo_mini",
            "apollo",
            "torchao_adamw8bit",
            "torchao_adamw4bit",
            "torchao_cpu_offload_adamw",
        ],
        default=None,
    )
    parser.add_argument("--precision", choices=["amp", "transformer_engine_fp8"], default=None)
    parser.add_argument("--activation-offload", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    config = AsterConfig.from_yaml(args.model)
    train = TrainConfig.from_yaml(args.train_config)
    if args.sequence is not None:
        train.sequence_length = args.sequence
    if args.batch is not None:
        train.micro_batch_size = args.batch
    if args.accum is not None:
        train.gradient_accumulation_steps = args.accum
    if args.device is not None:
        train.device = args.device
    if args.optimizer is not None:
        train.optimizer = args.optimizer
    if args.precision is not None:
        train.precision_backend = args.precision
        config.linear_backend = "transformer_engine" if args.precision == "transformer_engine_fp8" else "torch"
    if args.activation_offload:
        train.activation_offload = True
    if args.compile:
        train.compile = True
    config.max_seq_len = max(config.max_seq_len, train.sequence_length)

    result: dict[str, Any] = {
        "status": "started",
        "model_config": args.model,
        "train_config": args.train_config,
        "resolved_model": config.to_dict(),
        "resolved_train": train.to_dict(),
        "steps": [],
    }
    output_path = Path(args.json) if args.json else None

    try:
        if train.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        device = torch.device(train.device)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision(train.matmul_precision)
        result["system"] = static_system_manifest(device)
        sampler = SystemSampler(device, min_interval=0.1)

        model = AsterLM(config)
        dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[train.dtype]
        if device.type == "cuda" and dtype != torch.float32:
            model = model.to(device=device, dtype=dtype)
            for name, parameter in model.named_parameters():
                if name.endswith(("A_log", "dt_bias")):
                    parameter.data = parameter.data.float()
        else:
            model = model.to(device)
        model.train()
        precision = PrecisionManager(train, device, dtype)
        result["architecture"] = model.architecture_summary()
        result["parameter_storage"] = parameter_storage(model)
        result["memory_after_model"] = cuda_snapshot(device)
        optimizer = build_optimizer(model, train)
        result["memory_after_optimizer_build"] = cuda_snapshot(device)

        if train.compile and config.linear_backend == "transformer_engine":
            raise ValueError("Do not combine --compile and Transformer Engine in the first probe")
        forward_model = (
            torch.compile(model, mode=train.compile_mode, dynamic=False) if train.compile else model
        )
        durations: list[float] = []
        tokens_per_step = train.sequence_length * train.micro_batch_size * train.gradient_accumulation_steps
        total_iterations = args.warmup + args.steps
        for iteration in range(total_iterations):
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            loss_value = 0.0
            for _ in range(train.gradient_accumulation_steps):
                ids = torch.randint(
                    0,
                    config.vocab_size,
                    (train.micro_batch_size, train.sequence_length),
                    device=device,
                )
                labels = torch.randint(0, config.vocab_size, ids.shape, device=device)
                with precision.activation_context():
                    with precision.forward_context():
                        output = forward_model(ids, labels=labels, return_logits=False)
                        loss = output.loss / train.gradient_accumulation_steps
                loss.backward()
                loss_value += float(output.loss.detach())
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train.max_grad_norm)
            optimizer.step()
            balance = model.update_moe_router_biases()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            duration = time.perf_counter() - started
            record = {
                "iteration": iteration + 1,
                "warmup": iteration < args.warmup,
                "seconds": duration,
                "tokens_per_second": tokens_per_step / duration,
                "loss": loss_value / train.gradient_accumulation_steps,
                "grad_norm": float(grad_norm),
                "memory": cuda_snapshot(device),
                "system": sampler.sample(force=True),
                **balance,
            }
            result["steps"].append(record)
            print(json.dumps(record, sort_keys=True))
            if iteration >= args.warmup:
                durations.append(duration)

        median = sorted(durations)[len(durations) // 2]
        result["summary"] = {
            "median_seconds": median,
            "median_tokens_per_second": tokens_per_step / median,
            "final_memory": cuda_snapshot(device),
            "fits_11p25_gib_peak": cuda_snapshot(device).get("peak_allocated_gib", 0) <= 11.25,
        }
        result["status"] = "ok"
    except torch.cuda.OutOfMemoryError as exc:
        result["status"] = "oom"
        result["error"] = str(exc)
        if "device" in locals():
            result["memory_at_failure"] = cuda_snapshot(device)
        print(f"CUDA OOM: {exc}")
    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        print(result["error"])
    finally:
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
            print(f"wrote {output_path}")

    if result["status"] != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
