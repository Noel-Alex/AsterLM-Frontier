#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from asterlm.data import format_chat
from asterlm.generation import (
    GenerationConfig,
    download_hub_checkpoint,
    generate,
    generate_mtp_greedy,
    list_hub_checkpoints,
    load_runtime,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with an AsterLM checkpoint")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint")
    source.add_argument("--hub-repo", help="Hugging Face model repository")
    parser.add_argument("--hub-run", default=None, help="Run folder under runs/ in the Hub repo")
    parser.add_argument(
        "--hub-selector",
        default="latest",
        help="latest, final, tokens:N, or an exact checkpoint directory name",
    )
    parser.add_argument("--hub-revision", default="main")
    parser.add_argument("--hub-cache-dir", default=None)
    parser.add_argument("--list-hub-checkpoints", action="store_true")
    parser.add_argument("--tokenizer", default=None)
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
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--quantization", choices=["none", "int4", "int8"], default="none")
    parser.add_argument("--mtp-greedy", action="store_true", help="Use exact MTP reference speculation")
    parser.add_argument("--cache-dtype", choices=["bfloat16", "float8", "int8", "int4", "hadamard_int4"], default=None)
    parser.add_argument("--cache-recent-tokens", type=int, default=None)
    parser.add_argument("--cache-quantize-rope", action="store_true")
    args = parser.parse_args()

    if args.hub_repo and args.list_hub_checkpoints:
        import json

        print(json.dumps(list_hub_checkpoints(args.hub_repo, revision=args.hub_revision), indent=2))
        return
    checkpoint = args.checkpoint
    if args.hub_repo:
        checkpoint, identity = download_hub_checkpoint(
            args.hub_repo,
            run=args.hub_run,
            selector=args.hub_selector,
            revision=args.hub_revision,
            cache_dir=args.hub_cache_dir,
        )
        print(f"Loaded Hub checkpoint: {identity}")
    assert checkpoint is not None
    tokenizer_path = args.tokenizer
    bundled_tokenizer = Path(checkpoint) / "tokenizer.json"
    if tokenizer_path is None:
        tokenizer_path = str(bundled_tokenizer if bundled_tokenizer.is_file() else "artifacts/tokenizer.json")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    model, tokenizer = load_runtime(
        checkpoint,
        tokenizer_path,
        args.model,
        device=device,
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
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
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
