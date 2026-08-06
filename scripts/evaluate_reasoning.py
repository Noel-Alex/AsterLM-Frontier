#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from asterlm.generation.decode import load_runtime
from asterlm.reasoning import RLVRConfig, ReasoningMode, format_reasoning_prompt, score_completion
from asterlm.reasoning.io import iter_json_records
from asterlm.reasoning.rollout import sample_rollout


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate pass@1/pass@k reasoning with deterministic verifiers")
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reasoning", default="configs/reasoning/rlvr_laptop_gspo.yaml")
    parser.add_argument("--data", default="data/reasoning/eval_prompts.jsonl")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default="runs/reasoning-eval/results.jsonl")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = RLVRConfig.from_yaml(args.reasoning)
    records = list(iter_json_records(args.data))
    if args.limit:
        records = records[: args.limit]
    model, tokenizer = load_runtime(args.checkpoint, config.tokenizer_path, model_config=args.model, device=args.device)
    eos = [tokenizer.token_to_id("<|end|>"), tokenizer.token_to_id("<|endoftext|>")]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    solved = 0
    total_samples = 0
    with output.open("w", encoding="utf-8") as handle:
        for item_index, record in enumerate(records):
            prompt = format_reasoning_prompt(str(record["prompt"]), ReasoningMode.THINK)
            prompt_ids = tokenizer.encode(prompt)[-config.max_prompt_tokens :]
            tensor = torch.tensor(prompt_ids, dtype=torch.long, device=args.device).unsqueeze(0)
            attempts = []
            any_correct = False
            for sample_index in range(args.samples):
                rollout = sample_rollout(
                    model,
                    tensor,
                    max_new_tokens=config.max_completion_tokens,
                    min_new_tokens=config.min_completion_tokens,
                    temperature=config.temperature if args.samples > 1 else 0.01,
                    top_k=config.top_k,
                    top_p=config.top_p,
                    min_p=config.min_p,
                    repetition_penalty=config.repetition_penalty,
                    eos_token_ids=eos,
                    seed=config.seed + item_index * 1009 + sample_index,
                    prefill_chunk_size=config.prefill_chunk_size,
                )
                text = tokenizer.decode(rollout.completion_ids, skip_special_tokens=False)
                enriched = {**record, "completion_tokens": len(rollout.completion_ids)}
                reward = score_completion(enriched, text, config)
                any_correct = any_correct or bool(reward.correctness)
                attempts.append({"completion": text, "reward": reward.to_dict()})
                total_samples += 1
            solved += int(any_correct)
            handle.write(json.dumps({**record, "attempts": attempts, "pass": any_correct}, ensure_ascii=False) + "\n")
            print(f"{item_index + 1}/{len(records)} pass@{args.samples}={solved / (item_index + 1):.3f}")
    print(json.dumps({"problems": len(records), "samples": total_samples, f"pass@{args.samples}": solved / max(1, len(records))}, indent=2))


if __name__ == "__main__":
    main()
