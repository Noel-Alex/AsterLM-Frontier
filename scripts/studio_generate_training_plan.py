#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]

BASE = {
    "stage1": ROOT / "configs/train/frontier_100b_stage1_8k.yaml",
    "stage2": ROOT / "configs/train/frontier_100b_stage2_16k.yaml",
    "stage3": ROOT / "configs/train/frontier_100b_stage3_32k.yaml",
}


def load_train(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if "train" not in raw:
        raise ValueError(f"{path} has no train: mapping")
    return raw


def rolling_save_steps(train: dict[str, Any], checkpoint_tokens: int) -> int:
    cfg = train["train"]
    per_step = (
        int(cfg["sequence_length"])
        * int(cfg.get("micro_batch_size", 1))
        * int(cfg.get("gradient_accumulation_steps", 1))
    )
    return max(1, round(checkpoint_tokens / max(1, per_step)))


def set_token_horizon(raw: dict[str, Any], tokens: int, checkpoint_tokens: int) -> None:
    cfg = raw["train"]
    cfg["max_tokens"] = int(tokens)
    per_step = (
        int(cfg["sequence_length"])
        * int(cfg.get("micro_batch_size", 1))
        * int(cfg.get("gradient_accumulation_steps", 1))
    )
    effective_steps = max(1, math.ceil(tokens / max(1, per_step)))
    cfg["max_steps"] = max(int(cfg.get("max_steps", 1)), effective_steps + 10)
    # Keep stock warmups for long runs, but make generated configs valid for
    # smaller Studio experiments too.
    cfg["warmup_steps"] = min(
        int(cfg.get("warmup_steps", 0)),
        max(0, effective_steps // 20),
    )
    cfg["save_interval"] = rolling_save_steps(raw, checkpoint_tokens)


def unique_sorted(values: list[int], ceiling: int) -> list[int]:
    return sorted({int(value) for value in values if 0 < int(value) <= ceiling})


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a token-honest three-stage Aster pretraining schedule")
    parser.add_argument("--name", required=True, help="Run/config stem, e.g. aster-84b")
    parser.add_argument("--tokens", type=int, required=True, help="Total intended training tokens")
    parser.add_argument("--data", required=True, help="Generated clean DataConfig path")
    parser.add_argument("--output-dir", default="configs/studio/train")
    parser.add_argument("--checkpoint-tokens", type=int, default=25_000_000)
    parser.add_argument("--keep-last", type=int, default=6)
    parser.add_argument("--stage1-fraction", type=float, default=0.92)
    parser.add_argument("--stage2-fraction", type=float, default=0.06)
    parser.add_argument("--allow-repeat", action="store_true")
    parser.add_argument("--available-tokens", type=int, default=None)
    args = parser.parse_args()

    if args.tokens <= 0:
        raise SystemExit("--tokens must be > 0")
    if args.available_tokens is not None and args.tokens > args.available_tokens and not args.allow_repeat:
        raise SystemExit(
            f"Requested {args.tokens:,} training tokens but only {args.available_tokens:,} are declared available. "
            "Refusing implicit repetition; pass --allow-repeat only when repetition is intentional."
        )
    if args.stage1_fraction <= 0 or args.stage2_fraction < 0:
        raise SystemExit("Invalid stage fractions")
    if args.stage1_fraction + args.stage2_fraction >= 1:
        raise SystemExit("stage1 + stage2 fractions must leave a positive stage3 fraction")

    total = int(args.tokens)
    stage1_tokens = int(round(total * args.stage1_fraction))
    stage2_tokens = int(round(total * args.stage2_fraction))
    stage3_tokens = total - stage1_tokens - stage2_tokens

    configs = {
        "stage1": load_train(BASE["stage1"]),
        "stage2": load_train(BASE["stage2"]),
        "stage3": load_train(BASE["stage3"]),
    }
    horizons = {
        "stage1": stage1_tokens,
        "stage2": stage2_tokens,
        "stage3": stage3_tokens,
    }

    out_root = ROOT / args.output_dir
    out_root.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}

    for index, key in enumerate(("stage1", "stage2", "stage3"), start=1):
        raw = configs[key]
        tokens = horizons[key]
        set_token_horizon(raw, tokens, args.checkpoint_tokens)
        cfg = raw["train"]

        context = int(cfg["sequence_length"])
        run_dir = f"runs/{args.name}-stage{index}-{context // 1024}k"
        cfg["output_dir"] = run_dir
        cfg["wandb_run_name"] = f"{args.name}-stage{index}-{context // 1024}k"
        cfg["keep_last_checkpoints"] = max(1, int(args.keep_last))

        if key == "stage1":
            candidate = [
                int(stage1_tokens * 0.10),
                18_400_000_000,
                int(stage1_tokens * 0.50),
                50_000_000_000,
                int(stage1_tokens * 0.75),
                stage1_tokens,
            ]
        else:
            candidate = [tokens // 2, tokens]
        cfg["milestone_tokens"] = unique_sorted(candidate, tokens)

        target = out_root / f"{args.name}_stage{index}_{context // 1024}k.yaml"
        target.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        outputs[key] = str(target.relative_to(ROOT))

    repetition = (
        float(total) / float(args.available_tokens)
        if args.available_tokens and args.available_tokens > 0
        else None
    )

    plan = {
        "version": 1,
        "name": args.name,
        "data_config": args.data,
        "total_tokens": total,
        "available_tokens": args.available_tokens,
        "repetition_factor": repetition,
        "checkpoint_tokens": args.checkpoint_tokens,
        "stages": [
            {
                "id": "stage1",
                "context": configs["stage1"]["train"]["sequence_length"],
                "tokens": stage1_tokens,
                "train_config": outputs["stage1"],
                "output_dir": configs["stage1"]["train"]["output_dir"],
                "init_from": None,
            },
            {
                "id": "stage2",
                "context": configs["stage2"]["train"]["sequence_length"],
                "tokens": stage2_tokens,
                "train_config": outputs["stage2"],
                "output_dir": configs["stage2"]["train"]["output_dir"],
                "init_from": configs["stage1"]["train"]["output_dir"],
            },
            {
                "id": "stage3",
                "context": configs["stage3"]["train"]["sequence_length"],
                "tokens": stage3_tokens,
                "train_config": outputs["stage3"],
                "output_dir": configs["stage3"]["train"]["output_dir"],
                "init_from": configs["stage2"]["train"]["output_dir"],
            },
        ],
    }
    plan_path = out_root / f"{args.name}_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
