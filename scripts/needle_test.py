#!/usr/bin/env python
from __future__ import annotations

import argparse
import random

import torch

from asterlm.generation import GenerationConfig, generate, load_runtime


def make_context(tokenizer, target_tokens: int, depth: float, key: str, value: str) -> str:
    filler = (
        "A field notebook records ordinary observations about weather, roads, books, tools, and gardens. "
        "Each entry is independent and should be read carefully.\n"
    )
    needle = f"IMPORTANT RECORD: the value associated with {key} is {value}. Remember it exactly.\n"
    pieces = []
    while len(tokenizer.encode("".join(pieces))) < target_tokens:
        pieces.append(filler)
    text = "".join(pieces)
    split = int(len(text) * depth)
    return text[:split] + needle + text[split:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple long-context needle retrieval evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lengths", default="1024,2048,4096,8192")
    parser.add_argument("--depths", default="0.1,0.5,0.9")
    args = parser.parse_args()
    model, tokenizer = load_runtime(args.checkpoint, args.tokenizer, args.model, args.device)
    lengths = [int(x) for x in args.lengths.split(",")]
    depths = [float(x) for x in args.depths.split(",")]
    rng = random.Random(2026)
    passed = 0
    total = 0
    for length in lengths:
        for depth in depths:
            key = f"ZX-{rng.randrange(100000, 999999)}"
            value = str(rng.randrange(10000000, 99999999))
            context = make_context(tokenizer, length, depth, key, value)
            prompt = context + f"\nQuestion: What is the exact value associated with {key}? Answer with only the value.\nAnswer:"
            ids = tokenizer.encode(prompt)
            if len(ids) > model.config.max_seq_len:
                print(f"skip length={length}: tokenized prompt {len(ids)} exceeds max_seq_len")
                continue
            input_ids = torch.tensor([ids], dtype=torch.long, device=args.device)
            result = generate(
                model,
                input_ids,
                GenerationConfig(max_new_tokens=24, temperature=0, eos_token_id=tokenizer.token_to_id("<|end|>")),
            )
            answer = tokenizer.decode(result[0, input_ids.shape[1] :].tolist()).strip()
            success = value in answer
            total += 1
            passed += int(success)
            print(f"length={length:5d} depth={depth:.1f} pass={success} expected={value} got={answer!r}")
    print(f"needle retrieval: {passed}/{total} ({passed / max(total, 1):.1%})")


if __name__ == "__main__":
    main()
