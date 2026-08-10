#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import torch
from torch import nn

from asterlm import TrainConfig
from asterlm.layers.ffn import SwiGLU
from asterlm.layers.moe_grouped_te import TEGroupedRoutedExperts
from asterlm.training.precision import PrecisionManager


def event() -> torch.cuda.Event:
    return torch.cuda.Event(enable_timing=True)


def median_by_key(samples: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: statistics.median(sample[key] for sample in samples)
        for key in sorted(samples[0])
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Split Aster's grouped-MoE dispatch/MLP/combine cost")
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--hidden", type=int, default=704)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.rows <= 0 or args.top_k <= 0 or args.top_k > args.experts:
        raise SystemExit("Invalid rows/top-k/expert shape")

    device = torch.device("cuda")
    torch.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)
    experts = nn.ModuleList(
        [
            SwiGLU(
                args.dim,
                args.hidden,
                dropout=0.0,
                linear_backend="transformer_engine",
            )
            for _ in range(args.experts)
        ]
    ).to(device=device, dtype=torch.bfloat16)
    bridge = TEGroupedRoutedExperts(
        experts,
        dim=args.dim,
        expert_hidden=args.hidden,
        num_experts=args.experts,
        align=16,
    )

    token_ids = torch.arange(args.rows, device=device, dtype=torch.long)
    top_idx = torch.stack(
        [(token_ids + slot) % args.experts for slot in range(args.top_k)],
        dim=1,
    )
    raw_weight = torch.randn(args.rows, args.top_k, device=device, dtype=torch.float32)
    top_weight = torch.softmax(raw_weight, dim=-1)
    source = torch.randn(args.rows, args.dim, device=device, dtype=torch.bfloat16)

    train = TrainConfig()
    train.precision_backend = "transformer_engine_fp8"
    train.fp8_recipe = "delayed"
    train.fp8_amax_history_len = 16
    precision = PrecisionManager(train, device, torch.bfloat16)

    samples: list[dict[str, float]] = []
    for iteration in range(args.warmup + args.steps):
        for parameter in experts.parameters():
            parameter.grad = None
        flat = source.detach().clone().requires_grad_(True)
        markers = [event() for _ in range(5)]
        markers[0].record()
        with precision.forward_context():
            packed, split_sizes, real_positions, sorted_token_idx, sorted_weight = (
                bridge._dispatch_and_pad(flat, top_idx, top_weight)
            )
            markers[1].record()
            expert_output_padded = bridge.fused(packed, split_sizes, split_sizes)
            markers[2].record()
            expert_output = expert_output_padded.index_select(0, real_positions)
            weighted = expert_output * sorted_weight.to(expert_output.dtype).unsqueeze(-1)
            routed_out = torch.zeros_like(flat)
            routed_out.index_add_(0, sorted_token_idx, weighted)
            markers[3].record()
        routed_out.float().square().mean().backward()
        markers[4].record()
        torch.cuda.synchronize(device)
        if iteration >= args.warmup:
            samples.append(
                {
                    "dispatch_pack_ms": markers[0].elapsed_time(markers[1]),
                    "grouped_mlp_forward_ms": markers[1].elapsed_time(markers[2]),
                    "combine_ms": markers[2].elapsed_time(markers[3]),
                    "backward_ms": markers[3].elapsed_time(markers[4]),
                    "forward_total_ms": markers[0].elapsed_time(markers[3]),
                    "forward_backward_total_ms": markers[0].elapsed_time(markers[4]),
                }
            )

    medians = median_by_key(samples)
    forward_total = medians["forward_total_ms"]
    result: dict[str, Any] = {
        "schema_version": 1,
        "device": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "shape": {
            "rows": args.rows,
            "dim": args.dim,
            "hidden": args.hidden,
            "experts": args.experts,
            "top_k": args.top_k,
            "assignments": args.rows * args.top_k,
        },
        "precision": "transformer_engine_delayed_fp8",
        "warmup": args.warmup,
        "steps": args.steps,
        "median_ms": medians,
        "forward_share_percent": {
            "dispatch_pack": 100.0 * medians["dispatch_pack_ms"] / forward_total,
            "grouped_mlp": 100.0 * medians["grouped_mlp_forward_ms"] / forward_total,
            "combine": 100.0 * medians["combine_ms"] / forward_total,
        },
        "samples_ms": samples,
    }
    print(json.dumps(result, indent=2))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.json.with_suffix(args.json.suffix + ".partial")
        temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.json)


if __name__ == "__main__":
    main()
