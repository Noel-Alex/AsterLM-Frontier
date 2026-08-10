#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from asterlm import TrainConfig
from asterlm.layers.ffn import SwiGLU as AsterSwiGLU
from asterlm.training.precision import PrecisionManager


class GroupedSharedExpert:
    """Existing TE grouped-ops path with one group and authoritative Aster weights."""

    def __init__(self, expert: AsterSwiGLU, dim: int, hidden: int) -> None:
        from transformer_engine.pytorch.ops import GroupedLinear, Sequential, SwiGLU

        self.gate_up = GroupedLinear(
            num_groups=1,
            in_features=dim,
            out_features=2 * hidden,
            bias=False,
            dtype=torch.float32,
            device="meta",
        )
        self.down = GroupedLinear(
            num_groups=1,
            in_features=hidden,
            out_features=dim,
            bias=False,
            dtype=torch.float32,
            device="meta",
        )
        self.gate_up.weight0 = expert.gate_up.weight
        self.down.weight0 = expert.down.weight
        self.pipeline = Sequential(self.gate_up, SwiGLU(), self.down)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] % 16:
            raise ValueError("Probe rows must be divisible by 16")
        split_sizes = torch.full((1,), x.shape[0], dtype=torch.int32, device=x.device)
        return self.pipeline(x, split_sizes, split_sizes)


def zero_grad(parameters: list[torch.nn.Parameter]) -> None:
    for parameter in parameters:
        parameter.grad = None


def parity_report(
    baseline: Callable[[torch.Tensor], torch.Tensor],
    grouped: Callable[[torch.Tensor], torch.Tensor],
    source: torch.Tensor,
    parameters: list[torch.nn.Parameter],
) -> dict[str, float]:
    def run(fn: Callable[[torch.Tensor], torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        zero_grad(parameters)
        x = source.detach().clone().requires_grad_(True)
        output = fn(x)
        output.float().square().mean().backward()
        return output.detach(), x.grad.detach().clone(), [parameter.grad.detach().clone() for parameter in parameters]

    baseline_output, baseline_input_grad, baseline_weight_grads = run(baseline)
    grouped_output, grouped_input_grad, grouped_weight_grads = run(grouped)

    def relative_l2(left: torch.Tensor, right: torch.Tensor) -> float:
        return float((left.float() - right.float()).norm() / left.float().norm().clamp_min(1e-12))

    return {
        "output_max_abs": float((baseline_output.float() - grouped_output.float()).abs().max()),
        "output_relative_l2": relative_l2(baseline_output, grouped_output),
        "input_grad_relative_l2": relative_l2(baseline_input_grad, grouped_input_grad),
        "gate_up_grad_relative_l2": relative_l2(baseline_weight_grads[0], grouped_weight_grads[0]),
        "down_grad_relative_l2": relative_l2(baseline_weight_grads[1], grouped_weight_grads[1]),
    }


def benchmark(
    implementations: dict[str, Callable[[torch.Tensor], torch.Tensor]],
    source: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    precision: PrecisionManager,
    warmup: int,
    steps: int,
    repetitions: int,
) -> dict[str, Any]:
    samples: dict[str, list[float]] = {name: [] for name in implementations}
    names = list(implementations)
    orders = [names[offset:] + names[:offset] for offset in range(len(names))]
    for repetition in range(repetitions):
        for name in orders[repetition % len(orders)]:
            fn = implementations[name]
            for iteration in range(warmup + steps):
                zero_grad(parameters)
                x = source.detach().clone().requires_grad_(True)
                started = torch.cuda.Event(enable_timing=True)
                ended = torch.cuda.Event(enable_timing=True)
                started.record()
                with precision.forward_context():
                    output = fn(x)
                output.float().square().mean().backward()
                ended.record()
                torch.cuda.synchronize(source.device)
                if iteration >= warmup:
                    samples[name].append(started.elapsed_time(ended))
    medians = {name: statistics.median(values) for name, values in samples.items()}
    return {
        "warmup": warmup,
        "steps": steps,
        "repetitions": repetitions,
        "samples_ms": samples,
        "median_ms": medians,
        "grouped_speedup_percent": 100.0 * (medians["baseline"] / medians["grouped"] - 1.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe TE grouped shared-expert execution on this GPU")
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--hidden", type=int, default=704)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.rows <= 0 or args.rows % 16:
        raise SystemExit("--rows must be positive and divisible by 16")

    device = torch.device("cuda")
    torch.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)
    expert = AsterSwiGLU(
        args.dim,
        args.hidden,
        dropout=0.0,
        linear_backend="transformer_engine",
    ).to(device=device, dtype=torch.bfloat16)
    grouped = GroupedSharedExpert(expert, args.dim, args.hidden)
    parameters = [expert.gate_up.weight, expert.down.weight]
    source = torch.randn(args.rows, args.dim, device=device, dtype=torch.bfloat16)

    parity = parity_report(expert, grouped, source, parameters)
    train = TrainConfig()
    train.precision_backend = "transformer_engine_fp8"
    train.fp8_recipe = "delayed"
    train.fp8_amax_history_len = 16
    precision = PrecisionManager(train, device, torch.bfloat16)
    timing = benchmark(
        {"baseline": expert, "grouped": grouped},
        source,
        parameters,
        precision,
        args.warmup,
        args.steps,
        args.repetitions,
    )
    result = {
        "schema_version": 1,
        "device": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "shape": {"rows": args.rows, "dim": args.dim, "hidden": args.hidden},
        "precision": "transformer_engine_delayed_fp8",
        "bf16_parity": parity,
        "timing": timing,
    }
    print(json.dumps(result, indent=2))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.json.with_suffix(args.json.suffix + ".partial")
        temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.json)


if __name__ == "__main__":
    main()
