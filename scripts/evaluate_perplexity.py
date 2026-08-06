#!/usr/bin/env python
from __future__ import annotations

import argparse
import math

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from asterlm import DataConfig
from asterlm.data import AsterTokenizer, PackedTokenDataset
from asterlm.generation import load_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate held-out next-token loss/perplexity")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    parser.add_argument("--data", default="configs/data/pretrain_mixture_v1.yaml")
    parser.add_argument("--sequence", type=int, default=4096)
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    model, tokenizer = load_runtime(args.checkpoint, args.tokenizer, args.model, args.device)
    data = DataConfig.from_yaml(args.data)
    if not data.validation_sources:
        raise ValueError("The data config has no validation_sources")
    dataset = PackedTokenDataset(tokenizer, data, args.sequence, validation=True)
    loader = iter(DataLoader(dataset, batch_size=1, num_workers=0))
    losses = []
    main_losses = []
    with torch.inference_mode():
        for _ in tqdm(range(args.batches)):
            batch = {k: v.to(args.device) for k, v in next(loader).items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
                output = model(**batch)
            losses.append(float(output.loss))
            main_losses.append(float(output.main_loss))
    loss = sum(losses) / len(losses)
    main = sum(main_losses) / len(main_losses)
    print(f"total loss: {loss:.6f}")
    print(f"next-token loss: {main:.6f}")
    print(f"perplexity: {math.exp(min(main, 20)):.4f}")


if __name__ == "__main__":
    main()
