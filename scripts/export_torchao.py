#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from asterlm.generation import load_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a TorchAO FFN-weight-only inference artifact")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    parser.add_argument("--model", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["int4", "int8"], default="int4")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    model, _ = load_runtime(
        args.checkpoint,
        args.tokenizer,
        args.model,
        device=args.device,
        quantization=args.mode,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        from safetensors.torch import save_model

        save_model(model, str(output), metadata={"quantization": args.mode, "scope": "ffn+mtp"})
    except Exception as exc:
        raise RuntimeError(
            "This TorchAO build could not serialize its tensor subclasses. Use runtime --quantization instead."
        ) from exc
    print(f"Saved {args.mode} FFN/MTP quantized artifact to {output}")


if __name__ == "__main__":
    main()
