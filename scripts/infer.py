#!/usr/bin/env python
from __future__ import annotations

import argparse

import torch

from asterlm.data import format_chat
from asterlm.generation import GenerationConfig, generate, generate_mtp_greedy, load_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with an AsterLM checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    parser.add_argument("--model", default=None)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--system", default="You are a helpful, accurate assistant.")
    parser.add_argument("--raw", action="store_true", help="Do not wrap the prompt in the chat template")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--min-p", type=float, default=0.02)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--prefill-chunk-size", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--quantization", choices=["none", "int4", "int8"], default="none")
    parser.add_argument("--mtp-greedy", action="store_true", help="Use exact MTP reference speculation")
    parser.add_argument("--cache-dtype", choices=["bfloat16", "float8", "int8", "int4", "hadamard_int4"], default=None)
    parser.add_argument("--cache-recent-tokens", type=int, default=None)
    parser.add_argument("--cache-quantize-rope", action="store_true")
    args = parser.parse_args()

    model, tokenizer = load_runtime(
        args.checkpoint,
        args.tokenizer,
        args.model,
        device=args.device,
        compile_model=args.compile,
        quantization=args.quantization,
    )
    if args.cache_dtype is not None:
        model.config.cache_dtype = args.cache_dtype
    if args.cache_recent_tokens is not None:
        model.config.cache_recent_tokens = args.cache_recent_tokens
    if args.cache_quantize_rope:
        model.config.cache_quantize_rope = True
    if args.raw:
        prompt = args.prompt
    else:
        prompt = format_chat(
            [
                {"role": "system", "content": args.system},
                {"role": "user", "content": args.prompt},
            ],
            add_generation_prompt=True,
        )
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=args.device)
    eos = tokenizer.token_to_id("<|end|>")
    if args.mtp_greedy:
        result, stats = generate_mtp_greedy(model, ids, args.max_new_tokens, eos)
        print(tokenizer.decode(result[0, ids.shape[1] :].tolist()))
        print(
            f"\n[MTP rounds={stats.rounds}, accepted={stats.accepted}/{stats.drafted}, "
            f"rate={stats.acceptance_rate:.1%}]"
        )
    else:
        generation = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            min_p=args.min_p,
            repetition_penalty=args.repetition_penalty,
            eos_token_id=eos,
            prefill_chunk_size=args.prefill_chunk_size,
        )
        result = generate(model, ids, generation)
        print(tokenizer.decode(result[0, ids.shape[1] :].tolist()))


if __name__ == "__main__":
    main()
