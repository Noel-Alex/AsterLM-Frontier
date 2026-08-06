#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from asterlm.generation.decode import load_runtime
from asterlm.reasoning import RLVRConfig, ReasoningMode, format_reasoning_prompt
from asterlm.reasoning.io import atomic_write_jsonl, iter_json_records
from asterlm.reasoning.rollout import sample_rollout


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate exact on-policy reasoning rollouts")
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reasoning", default="configs/reasoning/rlvr_laptop_gspo.yaml")
    parser.add_argument("--prompts", default="data/reasoning/rlvr_prompts.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    config = RLVRConfig.from_yaml(args.reasoning)
    prompts = list(iter_json_records(args.prompts))
    if not prompts:
        raise ValueError("No RLVR prompts were found")
    rng = random.Random(config.seed + 100003 * args.iteration)
    candidates = min(
        len(prompts),
        config.prompts_per_iteration * config.prompt_oversample_factor,
    )
    selected = rng.sample(prompts, candidates) if candidates < len(prompts) else prompts[:]
    rng.shuffle(selected)

    model, tokenizer = load_runtime(
        args.checkpoint,
        config.tokenizer_path,
        model_config=args.model,
        device=args.device,
        compile_model=args.compile,
        quantization="none",
    )
    eos_ids = [tokenizer.token_to_id("<|end|>"), tokenizer.token_to_id("<|endoftext|>")]
    rows = []
    for group_id, prompt_record in enumerate(selected):
        mode = ReasoningMode.DIRECT if rng.random() < config.direct_mode_fraction else ReasoningMode.THINK
        prompt_text = format_reasoning_prompt(
            str(prompt_record["prompt"]),
            mode,
            force_open_tag=config.force_thinking_prefix,
        )
        prompt_ids = tokenizer.encode(prompt_text)
        if len(prompt_ids) > config.max_prompt_tokens:
            prompt_ids = prompt_ids[-config.max_prompt_tokens :]
        prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=args.device).unsqueeze(0)
        for sample_id in range(config.group_size):
            seed = config.seed + args.iteration * 1_000_003 + group_id * 1009 + sample_id
            rollout = sample_rollout(
                model,
                prompt_tensor,
                max_new_tokens=config.max_completion_tokens,
                min_new_tokens=config.min_completion_tokens,
                temperature=config.temperature,
                top_k=config.top_k,
                top_p=config.top_p,
                min_p=config.min_p,
                repetition_penalty=config.repetition_penalty,
                eos_token_ids=eos_ids,
                seed=seed,
                prefill_chunk_size=config.prefill_chunk_size,
            )
            rows.append(
                {
                    **prompt_record,
                    "iteration": args.iteration,
                    "group_id": group_id,
                    "sample_id": sample_id,
                    "mode": str(mode),
                    "prompt_text": prompt_text,
                    "prompt_ids": prompt_ids,
                    "completion_ids": rollout.completion_ids,
                    "old_token_logps": rollout.old_token_logps,
                    "completion": tokenizer.decode(rollout.completion_ids, skip_special_tokens=False),
                    "completion_tokens": len(rollout.completion_ids),
                    "finish_reason": rollout.finish_reason,
                    "rollout_seed": seed,
                }
            )
            print(
                f"iteration={args.iteration} group={group_id} sample={sample_id} "
                f"tokens={len(rollout.completion_ids)} finish={rollout.finish_reason}",
                flush=True,
            )
    count = atomic_write_jsonl(args.output, rows)
    print(json.dumps({"output": args.output, "rollouts": count}, indent=2))


if __name__ == "__main__":
    main()
