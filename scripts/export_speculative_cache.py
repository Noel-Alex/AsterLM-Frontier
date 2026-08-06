#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from asterlm.generation import load_runtime


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export compact target hidden/logit caches for EAGLE/DeepSpec-style draft experiments"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    parser.add_argument("--model", default=None)
    parser.add_argument("--input", required=True, help="JSONL containing a text field")
    parser.add_argument("--output", required=True)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--hidden-dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    model, tokenizer = load_runtime(
        args.checkpoint, args.tokenizer, args.model, device=args.device
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    index_path = output / "index.jsonl"
    hidden_dtype = torch.bfloat16 if args.hidden_dtype == "bfloat16" else torch.float16
    with Path(args.input).open("r", encoding="utf-8") as reader, index_path.open(
        "w", encoding="utf-8"
    ) as index:
        for row_id, line in enumerate(tqdm(reader, desc="target-cache")):
            if args.limit is not None and row_id >= args.limit:
                break
            record = json.loads(line)
            text = record.get(args.text_field)
            if not text:
                continue
            ids = tokenizer.encode(str(text))[: args.sequence_length]
            if len(ids) < 2:
                continue
            input_ids = torch.tensor([ids], dtype=torch.long, device=args.device)
            with torch.inference_mode():
                result = model(input_ids, return_hidden=True)
                assert result.logits is not None and result.hidden_states is not None
                values, indices = torch.topk(result.logits.float(), k=args.top_k, dim=-1)
            shard = output / f"sample-{row_id:08d}.pt"
            torch.save(
                {
                    "input_ids": input_ids.cpu(),
                    "hidden_states": result.hidden_states.to(hidden_dtype).cpu(),
                    "topk_indices": indices.to(torch.int32).cpu(),
                    "topk_logits": values.to(torch.float16).cpu(),
                    "checkpoint": str(args.checkpoint),
                },
                shard,
            )
            index.write(
                json.dumps(
                    {
                        "path": shard.name,
                        "tokens": len(ids),
                        "source_id": record.get("id", row_id),
                    }
                )
                + "\n"
            )
    print(f"Wrote compact draft-training cache to {output}")


if __name__ == "__main__":
    main()
