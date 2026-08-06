#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
from dataclasses import fields, is_dataclass
from typing import Any

import torch

from asterlm.generation import GenerationConfig, generate, load_runtime


def tensor_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(v) for v in value)
    if is_dataclass(value):
        return sum(tensor_bytes(getattr(value, f.name)) for f in fields(value))
    if hasattr(value, "states"):
        return tensor_bytes(value.states)
    return 0


def sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark AsterLM prefill, decode, and cache memory")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompt-tokens", type=int, default=2048)
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--quantization", choices=["none", "int4", "int8"], default="none")
    parser.add_argument("--cache-dtype", choices=["bfloat16", "float8", "int8", "int4", "hadamard_int4"], default=None)
    parser.add_argument("--cache-recent-tokens", type=int, default=None)
    parser.add_argument("--cache-quantize-rope", action="store_true")
    args = parser.parse_args()
    model, tokenizer = load_runtime(
        args.checkpoint, args.tokenizer, args.model, args.device, args.compile, args.quantization
    )
    if args.cache_dtype is not None:
        model.config.cache_dtype = args.cache_dtype
    if args.cache_recent_tokens is not None:
        model.config.cache_recent_tokens = args.cache_recent_tokens
    if args.cache_quantize_rope:
        model.config.cache_quantize_rope = True
    vocab = model.config.vocab_size
    prompt = torch.randint(7, vocab, (1, args.prompt_tokens), device=args.device)

    for _ in range(args.warmup):
        cache = model.make_cache()
        with torch.inference_mode():
            model(prompt[:, : min(128, args.prompt_tokens)], cache=cache, use_cache=True)
    sync(args.device)
    cache = model.make_cache()
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.inference_mode():
        output = model(prompt, cache=cache, use_cache=True)
    sync(args.device)
    prefill_s = time.perf_counter() - start

    token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(args.new_tokens):
            output = model(token, cache=cache, use_cache=True)
            token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
    sync(args.device)
    decode_s = time.perf_counter() - start

    print(model.architecture_summary())
    print(f"prefill: {args.prompt_tokens / prefill_s:,.0f} tokens/s ({prefill_s:.3f}s)")
    print(f"decode: {args.new_tokens / decode_s:,.2f} tokens/s ({decode_s:.3f}s)")
    print(f"cache storage: {cache.num_bytes / 2**20:.2f} MiB")
    if args.device.startswith("cuda"):
        print(f"peak CUDA allocation: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
