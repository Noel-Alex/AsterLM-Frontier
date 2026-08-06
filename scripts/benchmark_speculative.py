#!/usr/bin/env python
from __future__ import annotations

import argparse
import time

import torch

from asterlm.generation import GenerationConfig, generate, generate_mtp_greedy, load_runtime


def sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare exact cached greedy and reference MTP self-speculation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompt", default="Explain why the sky is blue in simple terms.")
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--confidence", type=float, default=0.0)
    args = parser.parse_args()

    model, tokenizer = load_runtime(args.checkpoint, args.tokenizer, args.model, args.device)
    prompt_ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=args.device)
    eos = tokenizer.token_to_id("<|endoftext|>")

    # Warm the recurrent and full-prefix paths once without including setup in timing.
    generate(model, prompt_ids, GenerationConfig(max_new_tokens=2, temperature=0.0, eos_token_id=None))
    generate_mtp_greedy(model, prompt_ids, max_new_tokens=2, eos_token_id=None)
    sync(args.device)

    start = time.perf_counter()
    greedy = generate(
        model,
        prompt_ids,
        GenerationConfig(max_new_tokens=args.new_tokens, temperature=0.0, eos_token_id=eos),
    )
    sync(args.device)
    greedy_s = time.perf_counter() - start

    start = time.perf_counter()
    speculative, stats = generate_mtp_greedy(
        model,
        prompt_ids,
        max_new_tokens=args.new_tokens,
        eos_token_id=eos,
        min_draft_confidence=args.confidence,
    )
    sync(args.device)
    speculative_s = time.perf_counter() - start

    greedy_new = greedy.shape[1] - prompt_ids.shape[1]
    speculative_new = speculative.shape[1] - prompt_ids.shape[1]
    exact = torch.equal(greedy, speculative)
    print(f"exact greedy match: {exact}")
    print(f"cached greedy: {greedy_new / max(greedy_s, 1e-9):.2f} tok/s ({greedy_s:.3f}s)")
    print(f"reference MTP: {speculative_new / max(speculative_s, 1e-9):.2f} tok/s ({speculative_s:.3f}s)")
    print(
        f"MTP rounds={stats.rounds}, drafted={stats.drafted}, accepted={stats.accepted}, "
        f"corrections={stats.corrections}, acceptance={stats.acceptance_rate:.2%}"
    )
    if not exact:
        raise SystemExit("Reference MTP output diverged from cached greedy output")
    print("Note: this full-prefix verifier measures acceptance/correctness, not production speedup.")


if __name__ == "__main__":
    main()
