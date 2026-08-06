#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import platform
import sys

import torch

from asterlm import AsterConfig, AsterLM


def gib(value: float) -> float:
    return value / 2**30


def cache_element_bytes(config: AsterConfig, tokens: int) -> float:
    hot = min(tokens, config.cache_recent_tokens)
    cold = max(0, tokens - hot)
    latent = config.latent_rank
    rope = config.rope_dim
    hot_bytes = hot * (latent + rope) * 2
    if config.cache_dtype == "bfloat16":
        cold_latent = cold * latent * 2
        cold_rope = cold * rope * 2
    elif config.cache_dtype == "float8":
        cold_latent = cold * latent
        cold_rope = cold * (1 if config.cache_quantize_rope else 2)
    else:
        bits = 8 if config.cache_dtype == "int8" else 4
        padded_latent = 1 << (latent - 1).bit_length() if config.cache_dtype == "hadamard_int4" else latent
        groups = math.ceil(padded_latent / config.cache_group_size)
        cold_latent = cold * (padded_latent * bits / 8 + groups * 2)
        if config.cache_quantize_rope:
            padded_rope = 1 << (rope - 1).bit_length() if config.cache_dtype == "hadamard_int4" else rope
            rope_groups = math.ceil(padded_rope / config.cache_group_size)
            cold_rope = cold * (padded_rope * bits / 8 + rope_groups * 2)
        else:
            cold_rope = cold * rope * 2
    return hot_bytes + cold_latent + cold_rope


def main() -> None:
    parser = argparse.ArgumentParser(description="Check AsterLM runtime and estimate memory")
    parser.add_argument("--model", default="configs/model/aster_moe_frontier_893m_a484m.yaml")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--sequence", type=int, default=8192)
    args = parser.parse_args()

    config = AsterConfig.from_yaml(args.model)
    original_backend = config.kda_backend
    if not importlib.util.find_spec("fla"):
        config.kda_backend = "torch"
    with torch.device("meta"):
        model = AsterLM(config)
    trainable_params = model.parameter_count()
    effective_params = model.effective_parameter_count()
    pattern = config.pattern
    latent_layers = pattern.count("latent")
    kda_layers = pattern.count("kda")
    cache_tokens = config.attention_window or args.sequence
    latent_cache = args.batch * latent_layers * cache_element_bytes(config, cache_tokens)
    kda_state = args.batch * kda_layers * config.n_heads * config.head_dim * config.head_dim * 2
    trainable_bytes_bf16 = sum(p.numel() * 2 for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())

    print("AsterLM Frontier system check")
    print("-" * 76)
    print(f"Python: {sys.version.split()[0]}")
    print(f"OS: {platform.platform()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"VRAM: {gib(props.total_memory):.2f} GiB")
        print(f"BF16 supported: {torch.cuda.is_bf16_supported()}")
        print(f"CUDA capability: {props.major}.{props.minor}")
    print(f"fla-core importable: {bool(importlib.util.find_spec('fla'))}")
    print(f"Transformer Engine importable: {bool(importlib.util.find_spec('transformer_engine'))}")
    print(f"torchao importable: {bool(importlib.util.find_spec('torchao'))}")
    print(f"Requested KDA backend: {original_backend}")
    print(f"Attention linear backend: {config.linear_backend}")
    print(f"FFN/expert backend: {config.ffn_backend}")
    print(f"Effective logical parameters: {effective_params / 1e6:.2f} M")
    print(f"Trainable tensor parameters: {trainable_params / 1e6:.2f} M")
    print(f"Active logical parameters/token: {model.active_parameter_count() / 1e6:.2f} M")
    print(f"BF16 trainable parameter storage: {gib(trainable_bytes_bf16):.2f} GiB")
    print(f"Packed/model buffers: {gib(buffer_bytes):.2f} GiB")
    print(f"Approx. persistent model before optimizer: {gib(trainable_bytes_bf16 + buffer_bytes):.2f} GiB")
    print(f"Approx. recurrent KDA state: {gib(kda_state):.4f} GiB")
    print(
        f"Configured latent cache ({cache_tokens:,} tokens, {config.cache_dtype}): "
        f"{gib(latent_cache):.4f} GiB"
    )
    print(f"Tokens/update: {args.batch * args.sequence:,} before gradient accumulation")
    if os.name == "nt" and original_backend != "torch":
        print("WARNING: native Windows is not the recommended FLA/Triton environment; use WSL2 or Linux.")
    if not importlib.util.find_spec("fla"):
        print("WARNING: the pure-PyTorch KDA fallback is intentionally slow. Install fla-core for real runs.")
    if config.ffn_backend == "loqt_int4":
        print("WARNING: LoQT-style INT4 is an experiment. Compare quality against BF16/FP8 before corpus training.")


if __name__ == "__main__":
    main()
