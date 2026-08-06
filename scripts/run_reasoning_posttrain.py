#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from asterlm.config import TrainConfig


def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def latest(output_dir: str) -> str | None:
    path = Path(output_dir) / "latest.txt"
    return path.read_text(encoding="utf-8").strip() if path.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end cold-start SFT and verifier-RL reasoning post-training")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--sft-train", default="configs/train/reasoning_sft_8k.yaml")
    parser.add_argument("--sft-data", default="configs/data/reasoning_mode_fusion_local.yaml")
    parser.add_argument("--rl-train", default="configs/train/reasoning_rl_laptop.yaml")
    parser.add_argument("--reasoning", default="configs/reasoning/rlvr_laptop_gspo.yaml")
    parser.add_argument("--reasoning-records", default="data/reasoning-frontier")
    parser.add_argument("--direct-records", nargs="*", default=[
        "data/posttrain-frontier/smol_smoltalk",
        "data/posttrain-frontier/smoltalk2_magpie",
        "data/posttrain-frontier/smoltalk2_multilingual",
        "data/posttrain-frontier/smoltalk2_science",
    ])
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--stop-after-sft", action="store_true")
    parser.add_argument("--rl-stop-after", type=int, default=None)
    args = parser.parse_args()

    if not args.skip_prepare:
        run([
            sys.executable,
            "scripts/prepare_reasoning_data.py",
            args.reasoning_records,
            "--sft-output", "data/reasoning/reasoning_sft.jsonl",
            "--rl-output", "data/reasoning/rlvr_prompts.jsonl",
            "--direct-fraction", "0.10",
        ])
        run([
            sys.executable,
            "scripts/prepare_direct_mode_data.py",
            *args.direct_records,
            "--output", "data/reasoning/direct_mode_sft.jsonl",
            "--max-records", "150000",
        ])

    sft_cfg = TrainConfig.from_yaml(args.sft_train)
    sft_checkpoint = latest(sft_cfg.output_dir)
    command = [
        sys.executable, "scripts/train_sft.py",
        "--model", args.model,
        "--train", args.sft_train,
        "--data", args.sft_data,
    ]
    if sft_checkpoint:
        command.extend(["--resume", sft_checkpoint])
    else:
        command.extend(["--checkpoint", args.base_checkpoint])
    run(command)
    sft_checkpoint = latest(sft_cfg.output_dir)
    if not sft_checkpoint:
        raise RuntimeError("Reasoning SFT produced no checkpoint")
    if args.stop_after_sft:
        print(json.dumps({"reasoning_sft_checkpoint": sft_checkpoint}, indent=2))
        return

    command = [
        sys.executable, "scripts/run_reasoning_rl.py",
        "--model", args.model,
        "--train", args.rl_train,
        "--reasoning", args.reasoning,
        "--checkpoint", sft_checkpoint,
        "--prompts", "data/reasoning/rlvr_prompts.jsonl",
    ]
    if args.rl_stop_after is not None:
        command.extend(["--stop-after", str(args.rl_stop_after)])
    run(command)


if __name__ == "__main__":
    main()
