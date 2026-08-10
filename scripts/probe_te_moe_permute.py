#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import nn

from asterlm import TrainConfig
from asterlm.layers.ffn import SwiGLU
from asterlm.layers.moe_grouped_te import TEGroupedRoutedExperts
from asterlm.training.precision import PrecisionManager


class TEPermuteGroupedExperts:
    """Experimental official TE permute/unpermute around Aster's grouped MLP."""

    def __init__(self, bridge: TEGroupedRoutedExperts) -> None:
        self.bridge = bridge

    def __call__(
        self,
        flat: torch.Tensor,
        top_idx: torch.Tensor,
        top_weight: torch.Tensor,
    ) -> torch.Tensor:
        from transformer_engine.pytorch import moe_permute, moe_unpermute

        num_tokens = flat.shape[0]
        top_k = top_idx.shape[1]
        num_assignments = num_tokens * top_k
        expert_ids = top_idx.reshape(-1).to(torch.long)
        counts = torch.bincount(expert_ids, minlength=self.bridge.num_experts).to(torch.long)
        padded_counts = ((counts + self.bridge.align - 1) // self.bridge.align) * self.bridge.align
        padded_counts = torch.where(
            padded_counts > 0,
            padded_counts,
            torch.full_like(padded_counts, self.bridge.align),
        )
        padding_per_expert = padded_counts - counts
        padding_before = torch.cumsum(padding_per_expert, dim=0) - padding_per_expert
        grouped_expert_ids = torch.repeat_interleave(
            torch.arange(self.bridge.num_experts, device=flat.device),
            counts,
            output_size=num_assignments,
        )
        real_positions = (
            torch.arange(num_assignments, device=flat.device, dtype=torch.long)
            + padding_before.index_select(0, grouped_expert_ids)
        )

        aligned_assignments = (
            (num_assignments + self.bridge.align - 1) // self.bridge.align
        ) * self.bridge.align
        capacity = aligned_assignments + self.bridge.num_experts * self.bridge.align
        split_sizes_long = padded_counts.clone()
        split_sizes_long[-1] += capacity - padded_counts.sum()

        permuted, row_id_map = moe_permute(
            flat,
            top_idx.to(torch.int32),
            num_assignments,
            max_token_num=num_assignments,
            map_type="index",
        )
        packed = flat.new_zeros((capacity, self.bridge.dim))
        packed.index_copy_(0, real_positions, permuted)
        split_sizes = split_sizes_long.to(torch.int32)
        expert_output_padded = self.bridge.fused(packed, split_sizes, split_sizes)
        expert_output = expert_output_padded.index_select(0, real_positions)
        return moe_unpermute(
            expert_output,
            row_id_map,
            merging_probs=top_weight.float(),
            restore_shape=flat.shape,
            map_type="index",
        )


def zero_grad(parameters: list[torch.nn.Parameter]) -> None:
    for parameter in parameters:
        parameter.grad = None


def parity_report(
    current: Callable[[torch.Tensor], torch.Tensor],
    candidate: Callable[[torch.Tensor], torch.Tensor],
    source: torch.Tensor,
    parameters: list[torch.nn.Parameter],
) -> dict[str, float]:
    def run(fn: Callable[[torch.Tensor], torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        zero_grad(parameters)
        x = source.detach().clone().requires_grad_(True)
        output = fn(x)
        output.float().square().mean().backward()
        return output.detach(), x.grad.detach().clone(), [parameter.grad.detach().clone() for parameter in parameters]

    current_output, current_input_grad, current_weight_grads = run(current)
    candidate_output, candidate_input_grad, candidate_weight_grads = run(candidate)

    def relative_l2(left: torch.Tensor, right: torch.Tensor) -> float:
        return float((left.float() - right.float()).norm() / left.float().norm().clamp_min(1e-12))

    weight_error = max(
        relative_l2(left, right)
        for left, right in zip(current_weight_grads, candidate_weight_grads, strict=True)
    )
    return {
        "output_max_abs": float((current_output.float() - candidate_output.float()).abs().max()),
        "output_relative_l2": relative_l2(current_output, candidate_output),
        "input_grad_relative_l2": relative_l2(current_input_grad, candidate_input_grad),
        "max_weight_grad_relative_l2": weight_error,
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
        "te_permute_speedup_percent": 100.0 * (medians["current"] / medians["te_permute"] - 1.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe official TE MoE permute/unpermute in Aster")
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--hidden", type=int, default=704)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda")
    torch.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)
    experts = nn.ModuleList(
        [SwiGLU(args.dim, args.hidden, 0.0, "transformer_engine") for _ in range(args.experts)]
    ).to(device=device, dtype=torch.bfloat16)
    bridge = TEGroupedRoutedExperts(
        experts,
        dim=args.dim,
        expert_hidden=args.hidden,
        num_experts=args.experts,
        align=16,
    )
    candidate = TEPermuteGroupedExperts(bridge)

    token_ids = torch.arange(args.rows, device=device, dtype=torch.long)
    top_idx = torch.stack(
        [(token_ids + slot) % args.experts for slot in range(args.top_k)],
        dim=1,
    )
    top_weight = torch.softmax(
        torch.randn(args.rows, args.top_k, device=device, dtype=torch.float32),
        dim=-1,
    )
    source = torch.randn(args.rows, args.dim, device=device, dtype=torch.bfloat16)
    parameters = list(experts.parameters())
    current_fn = lambda x: bridge(x, top_idx, top_weight)
    candidate_fn = lambda x: candidate(x, top_idx, top_weight)

    parity = parity_report(current_fn, candidate_fn, source, parameters)
    train = TrainConfig()
    train.precision_backend = "transformer_engine_fp8"
    train.fp8_recipe = "delayed"
    train.fp8_amax_history_len = 16
    precision = PrecisionManager(train, device, torch.bfloat16)
    timing = benchmark(
        {"current": current_fn, "te_permute": candidate_fn},
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
        "shape": {
            "rows": args.rows,
            "dim": args.dim,
            "hidden": args.hidden,
            "experts": args.experts,
            "top_k": args.top_k,
        },
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
