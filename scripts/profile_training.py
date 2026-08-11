#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

from asterlm import AsterConfig, AsterLM, TrainConfig
from asterlm.optim import build_optimizer
from asterlm.source_provenance import assert_expected_checkout_source
from asterlm.training.precision import PrecisionManager
from asterlm.training.telemetry import static_system_manifest


class ContinuousGpuSampler:
    """Sample GPU activity without synchronizing the training CUDA stream."""

    _FIELDS = (
        "utilization.gpu",
        "utilization.memory",
        "power.draw",
        "temperature.gpu",
        "clocks.current.sm",
        "clocks.current.memory",
        "memory.used",
    )

    def __init__(self, device_index: int, interval: float) -> None:
        self.device_index = int(device_index)
        self.interval = max(float(interval), 0.1)
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._phase = "setup"
        self._iteration = 0

    def set_phase(self, phase: str, iteration: int) -> None:
        self._phase = phase
        self._iteration = int(iteration)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="aster-gpu-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self.interval * 3))

    def _run(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            sample: dict[str, Any] = {
                "time_unix": time.time(),
                "phase": self._phase,
                "iteration": self._iteration,
            }
            try:
                output = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=" + ",".join(self._FIELDS),
                        "--format=csv,noheader,nounits",
                        "-i",
                        str(self.device_index),
                    ],
                    text=True,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                )
                values = [float(value.strip()) for value in output.splitlines()[0].split(",")]
                sample.update(dict(zip(self._FIELDS, values, strict=True)))
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                sample["error"] = f"{type(exc).__name__}: {exc}"
            self.samples.append(sample)
            self._stop.wait(max(0.0, self.interval - (time.monotonic() - started)))


def summarize_gpu_samples(samples: list[dict[str, Any]]) -> dict[str, float | int]:
    measured = [sample for sample in samples if sample.get("phase") == "measured"]
    summary: dict[str, float | int] = {"measured_sample_count": len(measured)}
    for field in ContinuousGpuSampler._FIELDS:
        values = [float(sample[field]) for sample in measured if field in sample]
        if values:
            summary[f"mean_{field.replace('.', '_')}"] = statistics.fmean(values)
            summary[f"median_{field.replace('.', '_')}"] = statistics.median(values)
            summary[f"p10_{field.replace('.', '_')}"] = sorted(values)[max(0, int(0.1 * (len(values) - 1)))]
            summary[f"p90_{field.replace('.', '_')}"] = sorted(values)[min(len(values) - 1, int(0.9 * (len(values) - 1)))]
    return summary


def gib(value: float) -> float:
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
        "parameter_tensors": 0,
        "buffer_gib": 0.0,
        "buffer_tensors": 0,
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
            result["parameter_tensors"] += 1
        else:
            buffer_total += size
            result["buffer_tensors"] += 1
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


@contextmanager
def measure_phase(
    name: str,
    *,
    device: torch.device,
    host_seconds: dict[str, float],
    cuda_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]],
):
    """Record host submission/synchronization time and default-stream CUDA time."""

    started = time.perf_counter()
    event_pair = None
    if device.type == "cuda":
        event_pair = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        event_pair[0].record()
    try:
        yield
    finally:
        if event_pair is not None:
            event_pair[1].record()
            cuda_events.setdefault(name, []).append(event_pair)
        host_seconds[name] = host_seconds.get(name, 0.0) + (time.perf_counter() - started)


def summarize_phase_timing(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for timing_kind in ("cuda_ms", "host_submit_or_sync_ms"):
        names = sorted(
            {
                name
                for record in records
                for name in record.get("phase_timing", {}).get(timing_kind, {})
            }
        )
        summary[timing_kind] = {
            name: statistics.median(
                float(record["phase_timing"][timing_kind][name])
                for record in records
                if name in record.get("phase_timing", {}).get(timing_kind, {})
            )
            for name in names
        }
    return summary


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
            "adamw",
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
    parser.add_argument(
        "--moe-implementation",
        choices=["reference", "grouped", "cutlass", "torch_grouped"],
        default=None,
        help="Select and record the physical expert implementation explicitly.",
    )
    parser.add_argument(
        "--apollo-disable-norm-limiter",
        action="store_true",
        help="Diagnostic only: disable APOLLO's norm-growth limiter for an explicit causal screen.",
    )
    parser.add_argument("--activation-offload", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--checkpoint-segment-size",
        type=int,
        default=None,
        help="Override the number of consecutive blocks inside each activation-checkpoint segment.",
    )
    parser.add_argument(
        "--disable-gradient-checkpointing",
        action="store_true",
        help="Execution-only control that preserves model math but retains all forward activations.",
    )
    parser.add_argument(
        "--allow-compile-transformer-engine-experimental",
        action="store_true",
        help=(
            "Allow the explicitly experimental torch.compile + Transformer Engine "
            "combination. This is disabled by default until numerical parity and "
            "end-to-end stability are established for the exact backend."
        ),
    )
    parser.add_argument("--gpu-sample-interval", type=float, default=0.5)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    source = assert_expected_checkout_source(repo_root)

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
    if args.apollo_disable_norm_limiter:
        train.apollo_disable_norm_limiter = True
    if args.activation_offload:
        train.activation_offload = True
    if args.compile:
        train.compile = True
    if args.checkpoint_segment_size is not None:
        if args.checkpoint_segment_size <= 0:
            raise ValueError("checkpoint segment size must be positive")
        config.checkpoint_segment_size = args.checkpoint_segment_size
    if args.disable_gradient_checkpointing:
        config.gradient_checkpointing = False
    config.max_seq_len = max(config.max_seq_len, train.sequence_length)

    result: dict[str, Any] = {
        "status": "started",
        "model_config": args.model,
        "train_config": args.train_config,
        "resolved_model": config.to_dict(),
        "resolved_train": train.to_dict(),
        "moe_implementation": args.moe_implementation or os.environ.get(
            "ASTER_MOE_IMPL", "reference"
        ),
        "source_provenance": source,
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

        # Deterministic profiling: seed model initialization explicitly.
        torch.manual_seed(train.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(train.seed)
        result["system"] = static_system_manifest(device)
        gpu_sampler = (
            ContinuousGpuSampler(device.index or 0, args.gpu_sample_interval)
            if device.type == "cuda"
            else None
        )

        model = AsterLM(config, moe_implementation=args.moe_implementation)
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

        if (
            train.compile
            and config.linear_backend == "transformer_engine"
            and not args.allow_compile_transformer_engine_experimental
        ):
            raise ValueError(
                "Compile + Transformer Engine remains experimental; pass "
                "--allow-compile-transformer-engine-experimental only for an explicit "
                "parity and performance probe"
            )
        forward_model = (
            torch.compile(model, mode=train.compile_mode, dynamic=False) if train.compile else model
        )
        durations: list[float] = []
        measured_phase_records: list[dict[str, Any]] = []
        # Keep synthetic benchmark data independent of model/backend RNG use.
        data_generator = torch.Generator(device=device)
        data_generator.manual_seed(train.seed + 100003)
        tokens_per_step = train.sequence_length * train.micro_batch_size * train.gradient_accumulation_steps
        total_iterations = args.warmup + args.steps
        if gpu_sampler is not None:
            gpu_sampler.start()
        for iteration in range(total_iterations):
            if gpu_sampler is not None:
                gpu_sampler.set_phase(
                    "warmup" if iteration < args.warmup else "measured",
                    iteration + 1,
                )
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            phase_host_seconds: dict[str, float] = {}
            phase_cuda_events: dict[
                str, list[tuple[torch.cuda.Event, torch.cuda.Event]]
            ] = {}
            loss_value = torch.zeros((), device=device, dtype=torch.float32)
            for _ in range(train.gradient_accumulation_steps):
                with measure_phase(
                    "synthetic_data",
                    device=device,
                    host_seconds=phase_host_seconds,
                    cuda_events=phase_cuda_events,
                ):
                    ids = torch.randint(
                        0,
                        config.vocab_size,
                        (train.micro_batch_size, train.sequence_length),
                        device=device,
                        generator=data_generator,
                    )
                    labels = torch.randint(
                        0,
                        config.vocab_size,
                        ids.shape,
                        device=device,
                        generator=data_generator,
                    )
                with measure_phase(
                    "forward",
                    device=device,
                    host_seconds=phase_host_seconds,
                    cuda_events=phase_cuda_events,
                ), precision.activation_context(), precision.forward_context():
                    output = forward_model(ids, labels=labels, return_logits=False)
                    if output.loss is None:
                        raise FloatingPointError("model returned no loss")
                    loss = output.loss / train.gradient_accumulation_steps
                with measure_phase(
                    "backward",
                    device=device,
                    host_seconds=phase_host_seconds,
                    cuda_events=phase_cuda_events,
                ):
                    loss.backward()
                loss_value.add_(output.loss.detach().float())
            with measure_phase(
                "clip_and_finite_gate",
                device=device,
                host_seconds=phase_host_seconds,
                cuda_events=phase_cuda_events,
            ):
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train.max_grad_norm)
                # The gradient gate catches any non-finite loss/backward before the
                # optimizer mutates parameters, while avoiding one GPU->CPU sync per
                # accumulation microbatch in the measured workload.
                if not torch.isfinite(grad_norm).all():
                    bad_grads = []
                    for name, parameter in model.named_parameters():
                        grad = parameter.grad
                        if grad is None:
                            continue
                        if not torch.isfinite(grad).all():
                            bad_grads.append(name)
                            if len(bad_grads) >= 8:
                                break
                    raise FloatingPointError(
                        "non-finite gradients before optimizer.step at "
                        f"iteration {iteration + 1}; grad_norm={float(grad_norm)}; "
                        f"first_bad_grad_tensors={bad_grads}"
                    )
            with measure_phase(
                "optimizer",
                device=device,
                host_seconds=phase_host_seconds,
                cuda_events=phase_cuda_events,
            ):
                optimizer.step()
            with measure_phase(
                "router_balance",
                device=device,
                host_seconds=phase_host_seconds,
                cuda_events=phase_cuda_events,
            ):
                balance = model.update_moe_router_biases()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            duration = time.perf_counter() - started
            phase_timing = {
                "cuda_ms": {
                    name: sum(start_event.elapsed_time(end_event) for start_event, end_event in pairs)
                    for name, pairs in phase_cuda_events.items()
                },
                # This includes Python submission and any synchronization incurred by
                # that phase. It is intentionally not described as GPU execution time.
                "host_submit_or_sync_ms": {
                    name: seconds * 1000.0 for name, seconds in phase_host_seconds.items()
                },
            }
            record = {
                "iteration": iteration + 1,
                "warmup": iteration < args.warmup,
                "seconds": duration,
                "tokens_per_second": tokens_per_step / duration,
                "loss": float(loss_value / train.gradient_accumulation_steps),
                "grad_norm": float(grad_norm),
                "memory": cuda_snapshot(device),
                "phase_timing": phase_timing,
                # Continuous sampling avoids launching another blocking nvidia-smi
                # query between every optimizer update.
                "system": (
                    dict(gpu_sampler.samples[-1])
                    if gpu_sampler is not None and gpu_sampler.samples
                    else {}
                ),
                **balance,
            }
            result["steps"].append(record)
            print(json.dumps(record, sort_keys=True))
            if iteration >= args.warmup:
                durations.append(duration)
                measured_phase_records.append(record)

        ordered_durations = sorted(durations)
        n_durations = len(ordered_durations)
        if n_durations == 0:
            raise RuntimeError("No measured profiling iterations were recorded")
        if n_durations % 2:
            median = ordered_durations[n_durations // 2]
        else:
            median = 0.5 * (
                ordered_durations[n_durations // 2 - 1]
                + ordered_durations[n_durations // 2]
            )
        result["summary"] = {
            "median_seconds": median,
            "median_tokens_per_second": tokens_per_step / median,
            "measured_wall_time_seconds": sum(durations),
            "measured_tokens": tokens_per_step * len(durations),
            "final_memory": cuda_snapshot(device),
            "fits_11p25_gib_peak": cuda_snapshot(device).get("peak_allocated_gib", 0) <= 11.25,
            "median_phase_timing": summarize_phase_timing(measured_phase_records),
        }
        if gpu_sampler is not None:
            gpu_sampler.stop()
            result["gpu_samples"] = gpu_sampler.samples
            result["summary"]["gpu"] = summarize_gpu_samples(gpu_sampler.samples)
            median_power = result["summary"]["gpu"].get("median_power_draw")
            if isinstance(median_power, (int, float)) and median_power > 0:
                result["summary"]["median_tokens_per_joule"] = (
                    result["summary"]["median_tokens_per_second"] / median_power
                )
            mean_power = result["summary"]["gpu"].get("mean_power_draw")
            measured_seconds = result["summary"]["measured_wall_time_seconds"]
            measured_tokens = result["summary"]["measured_tokens"]
            if isinstance(mean_power, (int, float)) and mean_power > 0:
                # `nvidia-smi` samples board power, so this is estimated GPU energy,
                # not whole-laptop wall energy. It is recorded for diagnostics and
                # research statistics only and never participates in promotion.
                energy_joules = mean_power * measured_seconds
                result["summary"]["estimated_gpu_energy_joules"] = energy_joules
                result["summary"]["estimated_gpu_energy_kwh"] = energy_joules / 3_600_000.0
                result["summary"]["estimated_gpu_joules_per_token"] = (
                    energy_joules / measured_tokens
                )
        result["status"] = "ok"
    except torch.cuda.OutOfMemoryError as exc:
        result["status"] = "oom"
        result["error"] = str(exc)
        if "device" in locals():
            result["memory_at_failure"] = cuda_snapshot(device)
        print(f"CUDA OOM: {exc}")
    except Exception as exc:  # noqa: BLE001 - profiler must persist arbitrary trial failures
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        print(result["error"])
    finally:
        if "gpu_sampler" in locals() and gpu_sampler is not None and gpu_sampler._thread is not None:
            gpu_sampler.stop()
            result.setdefault("gpu_samples", gpu_sampler.samples)
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
            print(f"wrote {output_path}")

    if result["status"] != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
