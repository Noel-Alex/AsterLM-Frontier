#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from asterlm import AsterConfig


def next_pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def quantized_vector_bytes(width: int, scheme: str, group_size: int) -> int:
    if scheme == "bfloat16":
        return 2 * width
    if scheme == "float8":
        return width
    if scheme not in {"int8", "int4", "hadamard_int4"}:
        raise ValueError(scheme)
    transformed = next_pow2(width) if scheme == "hadamard_int4" else width
    padded = math.ceil(transformed / group_size) * group_size
    groups = padded // group_size
    data = padded if scheme == "int8" else math.ceil(padded / 2)
    # Aster quantization stores one FP16 scale per group.
    return data + groups * 2


def layer_cache_bytes(config: AsterConfig, tokens: int, scheme: str) -> dict[str, int]:
    # Current Aster hot/cold cache: newest `recent_tokens + chunk_tokens` can stay
    # in BF16 because flushing happens only after the hysteresis threshold.
    if scheme == "bfloat16":
        hot = tokens
    else:
        hot = min(tokens, config.cache_recent_tokens + config.cache_chunk_tokens)
    cold = max(0, tokens - hot)

    hot_latent = hot * config.latent_rank * 2
    hot_rope = hot * config.rope_dim * 2
    cold_latent = cold * quantized_vector_bytes(config.latent_rank, scheme, config.cache_group_size)
    rope_scheme = scheme if config.cache_quantize_rope else "bfloat16"
    cold_rope = cold * quantized_vector_bytes(config.rope_dim, rope_scheme, config.cache_group_size)
    return {
        "hot_latent": hot_latent,
        "hot_rope": hot_rope,
        "cold_latent": cold_latent,
        "cold_rope": cold_rope,
        "total": hot_latent + hot_rope + cold_latent + cold_rope,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="configs/model/aster_moe_frontier_893m_fp8.yaml")
    parser.add_argument("--output", default="runs/frontier-vnext/cache_projection.json")
    parser.add_argument("--contexts", default="32768,131072,262144,1048576")
    args = parser.parse_args()

    config = AsterConfig.from_yaml(args.model)
    contexts = [int(x.strip()) for x in args.contexts.split(",") if x.strip()]
    latent_layers = config.pattern.count("latent")
    recurrent_layers = len(config.pattern) - latent_layers
    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "pattern": config.pattern,
        "latent_layers": latent_layers,
        "fixed_state_layers": recurrent_layers,
        "latent_rank": config.latent_rank,
        "rope_dim": config.rope_dim,
        "cache_group_size": config.cache_group_size,
        "cache_recent_tokens": config.cache_recent_tokens,
        "cache_chunk_tokens": config.cache_chunk_tokens,
        "cache_quantize_rope": config.cache_quantize_rope,
        "configured_attention_window": config.attention_window,
        "projection_note": (
            "These values project latent-attention KV storage only. KDA/GDN2 recurrent "
            "state is sequence-length independent. Actual decode latency can still scale "
            "badly if a global layer scans/dequantizes all cold chunks."
        ),
        "contexts": {},
    }

    for ctx in contexts:
        item: dict[str, Any] = {}
        for scheme in ["bfloat16", "float8", "int8", "int4", "hadamard_int4"]:
            one = layer_cache_bytes(config, ctx, scheme)
            total = one["total"] * latent_layers
            item[scheme] = {
                "one_latent_layer_gib": one["total"] / 2**30,
                "all_latent_layers_gib": total / 2**30,
                "bytes_per_token_all_latent_layers": total / max(ctx, 1),
                "breakdown_one_layer_bytes": one,
            }
        report["contexts"][str(ctx)] = item

    if config.attention_window is not None and max(contexts, default=0) > config.attention_window:
        report["important"] = (
            f"The current model config trims each latent cache to attention_window={config.attention_window:,}. "
            "A 1M-token inference experiment therefore requires an explicit larger/sparse global-cache policy; "
            "raising max_seq_len alone is not enough."
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Cache projection: {output}")
    for ctx in contexts:
        h4 = report["contexts"][str(ctx)]["hadamard_int4"]["all_latent_layers_gib"]
        bf = report["contexts"][str(ctx)]["bfloat16"]["all_latent_layers_gib"]
        print(f"  {ctx:>9,} tokens: BF16 {bf:6.3f} GiB | Hadamard INT4 {h4:6.3f} GiB")


if __name__ == "__main__":
    main()
