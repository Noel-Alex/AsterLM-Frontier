#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm

from asterlm.data.preference import encode_preference_sequence, pad_preference_batch, response_logprobs
from asterlm.generation import load_runtime
from asterlm.reasoning.io import iter_json_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute reference log-probabilities for memory-efficient DPO"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    parser.add_argument(
        "--input",
        required=True,
        help="JSON/JSONL(.gz/.zst) file or directory containing prompt/chosen/rejected records",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    source = Path(args.input)
    if not source.exists():
        raise FileNotFoundError(f"Preference input does not exist: {source}")

    model, tokenizer = load_runtime(args.checkpoint, args.tokenizer, args.model, args.device)
    if args.max_length > model.config.max_seq_len:
        raise ValueError("--max-length exceeds model max_seq_len")
    pad_id = tokenizer.token_to_id("<|pad|>")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output.with_suffix(output.suffix + ".tmp")
    written = 0
    try:
        with tmp_output.open("w", encoding="utf-8") as writer:
            for record in tqdm(iter_json_records(source), desc="Reference scoring", unit="pair"):
                missing = {key for key in ("prompt", "chosen", "rejected") if key not in record}
                if missing:
                    raise ValueError(
                        f"Preference record {written + 1} is missing required fields: {sorted(missing)}"
                    )
                chosen = encode_preference_sequence(
                    tokenizer, record["prompt"], record["chosen"], args.max_length
                )
                rejected = encode_preference_sequence(
                    tokenizer, record["prompt"], record["rejected"], args.max_length
                )
                input_ids, targets, masks = pad_preference_batch([chosen, rejected], pad_id)
                input_ids = input_ids.to(args.device)
                targets = targets.to(args.device)
                masks = masks.to(args.device)
                with torch.inference_mode(), torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=args.device.startswith("cuda"),
                ):
                    logits = model(input_ids).logits
                if logits is None:
                    raise RuntimeError("Reference scoring requires logits")
                logps, lengths = response_logprobs(logits, targets, masks)
                record["ref_chosen_logp"] = float(logps[0])
                record["ref_rejected_logp"] = float(logps[1])
                record["chosen_tokens"] = int(lengths[0])
                record["rejected_tokens"] = int(lengths[1])
                writer.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                written += 1
            writer.flush()
            os.fsync(writer.fileno())
        if written == 0:
            raise RuntimeError(f"No preference records found under {source}")
        os.replace(tmp_output, output)
    except BaseException:
        tmp_output.unlink(missing_ok=True)
        raise
    print(f"Wrote {written:,} reference-scored preference pairs to {output}")


if __name__ == "__main__":
    main()
