#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from asterlm import AsterConfig, AsterLM, TrainConfig
from asterlm.optim import build_optimizer
from asterlm.training.precision import PrecisionManager


def _event_value(event: Any, *names: str) -> float:
    for name in names:
        value = getattr(event, name, None)
        if value is not None:
            try:
                return float(value)
            except Exception:
                pass
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Aster vNext one-step operator/phase profiler")
    parser.add_argument("--model", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--sequence", type=int, default=1024)
    parser.add_argument("--optimizer", default=None)
    parser.add_argument("--precision", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"status": "started"}

    try:
        cfg = AsterConfig.from_yaml(args.model)
        train = TrainConfig.from_yaml(args.train)
        train.sequence_length = args.sequence
        if args.optimizer:
            train.optimizer = args.optimizer
        if args.precision:
            train.precision_backend = args.precision
            cfg.linear_backend = "transformer_engine" if args.precision == "transformer_engine_fp8" else "torch"
        cfg.max_seq_len = max(cfg.max_seq_len, args.sequence)

        device = torch.device(train.device)
        torch.manual_seed(train.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(train.seed)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision(train.matmul_precision)

        dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[train.dtype]
        model = AsterLM(cfg)
        if device.type == "cuda" and dtype != torch.float32:
            model = model.to(device=device, dtype=dtype)
            for name, parameter in model.named_parameters():
                if name.endswith(("A_log", "dt_bias")):
                    parameter.data = parameter.data.float()
        else:
            model = model.to(device)
        model.train()
        precision = PrecisionManager(train, device, dtype)
        optimizer = build_optimizer(model, train)

        data_gen = torch.Generator(device=device)
        data_gen.manual_seed(train.seed + 880301)

        def batch():
            ids = torch.randint(0, cfg.vocab_size, (1, args.sequence), device=device, generator=data_gen)
            labels = torch.randint(0, cfg.vocab_size, ids.shape, device=device, generator=data_gen)
            return ids, labels

        # Warm all lazy kernels/autotuners once outside the profiler.
        ids, labels = batch()
        with precision.forward_context():
            warm = model(ids, labels=labels, return_logits=False)
        warm.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        phase_wall: dict[str, float] = {}
        ids, labels = batch()
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as prof:
            t0 = time.perf_counter()
            with torch.profiler.record_function("ASTER::forward"):
                with precision.forward_context():
                    output = model(ids, labels=labels, return_logits=False)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            phase_wall["forward_s"] = time.perf_counter() - t0

            if not torch.isfinite(output.loss):
                raise FloatingPointError(f"non-finite profiled loss: {float(output.loss.detach())}")

            t0 = time.perf_counter()
            with torch.profiler.record_function("ASTER::backward"):
                output.loss.backward()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            phase_wall["backward_s"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            with torch.profiler.record_function("ASTER::optimizer"):
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            phase_wall["optimizer_s"] = time.perf_counter() - t0

        events = []
        for event in prof.key_averages(group_by_input_shape=False):
            events.append(
                {
                    "key": event.key,
                    "count": int(event.count),
                    "self_cpu_us": _event_value(event, "self_cpu_time_total"),
                    "cpu_us": _event_value(event, "cpu_time_total"),
                    "self_device_us": _event_value(event, "self_device_time_total", "self_cuda_time_total"),
                    "device_us": _event_value(event, "device_time_total", "cuda_time_total"),
                    "self_cpu_memory": int(_event_value(event, "self_cpu_memory_usage")),
                    "self_device_memory": int(_event_value(event, "self_device_memory_usage", "self_cuda_memory_usage")),
                }
            )
        events.sort(key=lambda item: item["self_device_us"], reverse=True)

        result.update(
            {
                "status": "ok",
                "sequence": args.sequence,
                "moe_impl": os.environ.get("ASTER_MOE_IMPL", "reference"),
                "loss": float(output.loss.detach()),
                "grad_norm_before_clip": float(grad_norm),
                "phase_wall": phase_wall,
                "peak_allocated_gib": (
                    torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
                ),
                "top_events": events[:120],
            }
        )
        table_path = out.with_suffix(".table.txt")
        table_path.write_text(
            prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=100),
            encoding="utf-8",
        )
    except Exception as exc:
        result.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
