#!/usr/bin/env python
from __future__ import annotations

import argparse

import torch

from asterlm import AsterConfig, AsterLM, TrainConfig
from asterlm.optim import build_hybrid_optimizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny CPU correctness smoke train")
    parser.add_argument("--steps", type=int, default=2, help="optimizer updates to run")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")

    torch.manual_seed(args.seed)
    config = AsterConfig(
        vocab_size=256,
        d_model=64,
        n_layers=4,
        n_heads=4,
        head_dim=16,
        ffn_hidden=176,
        max_seq_len=64,
        kda_ratio=1,
        kda_backend="torch",
        latent_rank=16,
        rope_dim=8,
        attention_window=32,
        sink_tokens=4,
        mtp_depth=2,
        mtp_rank=32,
        gradient_checkpointing=False,
    )
    model = AsterLM(config)
    optimizer = build_hybrid_optimizer(model, TrainConfig(device="cpu", max_steps=args.steps))
    for step in range(args.steps):
        input_ids = torch.randint(0, config.vocab_size, (2, 24))
        labels = torch.randint(0, config.vocab_size, (2, 24))
        output = model(input_ids, labels=labels)
        if output.loss is None:
            raise RuntimeError("model returned no loss")
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        print(f"step={step + 1} loss={float(output.loss.detach()):.4f}")
    print("smoke training passed", model.architecture_summary())


if __name__ == "__main__":
    main()
