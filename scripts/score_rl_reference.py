#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

import torch

from asterlm.generation.decode import load_runtime
from asterlm.reasoning.io import atomic_write_jsonl, iter_json_records
from asterlm.reasoning.losses import selected_token_logprobs_from_hidden
from asterlm.reasoning.training import build_rl_sequence, pad_rl_batch


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="Score rollouts under a frozen reference without co-resident models")
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True, help="Frozen SFT/reference checkpoint")
    parser.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--micro-batch-size", type=int, default=1)
    args = parser.parse_args()

    model, tokenizer = load_runtime(
        args.checkpoint,
        args.tokenizer,
        model_config=args.model,
        device=args.device,
        compile_model=False,
        quantization="none",
    )
    records = list(iter_json_records(args.input))
    pad_id = tokenizer.token_to_id("<|pad|>")
    for start in range(0, len(records), args.micro_batch_size):
        chunk = records[start : start + args.micro_batch_size]
        sequences = [build_rl_sequence(record) for record in chunk]
        batch = pad_rl_batch(sequences, pad_id)
        input_ids = batch["input_ids"].to(args.device)
        targets = batch["targets"].to(args.device)
        mask = batch["completion_mask"].to(args.device)
        model_output = model(input_ids, return_logits=False, return_hidden=True)
        hidden_states = model_output.hidden_states
        if hidden_states is None:
            raise RuntimeError("Reference model returned no hidden states")
        logps = selected_token_logprobs_from_hidden(
            model,
            hidden_states,
            targets,
            mask,
            chunk_size=model.config.lm_loss_chunk_size,
            checkpoint_chunks=False,
        )
        for local, record in enumerate(chunk):
            values = logps[local][mask[local]].detach().cpu().tolist()
            record["reference_token_logps"] = values
        print(f"reference-scored {min(start + len(chunk), len(records))}/{len(records)}", flush=True)
    atomic_write_jsonl(args.output, records)
    print(json.dumps({"output": args.output, "records": len(records)}, indent=2))


if __name__ == "__main__":
    main()
