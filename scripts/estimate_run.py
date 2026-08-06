#!/usr/bin/env python
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert measured throughput into a run plan")
    parser.add_argument("--tokens", type=float, required=True, help="Target training tokens")
    parser.add_argument("--tokens-per-second", type=float, required=True, help="Measured end-to-end throughput")
    parser.add_argument("--sequence", type=int, default=4096)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=32)
    args = parser.parse_args()
    seconds = args.tokens / args.tokens_per_second
    tokens_per_step = args.sequence * args.micro_batch * args.grad_accum
    steps = args.tokens / tokens_per_step
    print(f"Target: {args.tokens:,.0f} tokens")
    print(f"Optimizer updates: {steps:,.0f}")
    print(f"Wall-clock at measured throughput: {seconds / 3600:.1f} hours ({seconds / 86400:.2f} days)")
    print("Add at least 10-20% for evaluation, checkpointing, data stalls, and thermal throttling.")


if __name__ == "__main__":
    main()
