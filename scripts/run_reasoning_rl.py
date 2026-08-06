#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from asterlm.reasoning import RLVRConfig


def atomic_state(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def latest_checkpoint(root: Path) -> str:
    latest = root / "latest.txt"
    if not latest.exists():
        raise FileNotFoundError(f"No latest checkpoint at {latest}")
    return latest.read_text(encoding="utf-8").strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Crash-resumable single-GPU reasoning RL pipeline")
    parser.add_argument("--model", required=True)
    parser.add_argument("--train", default="configs/train/reasoning_rl_laptop.yaml")
    parser.add_argument("--reasoning", default="configs/reasoning/rlvr_laptop_gspo.yaml")
    parser.add_argument("--checkpoint", required=True, help="Cold-start reasoning SFT checkpoint")
    parser.add_argument("--prompts", default="data/reasoning/rlvr_prompts.jsonl")
    parser.add_argument("--reference-checkpoint", default=None)
    parser.add_argument("--start-iteration", type=int, default=None)
    parser.add_argument("--stop-after", type=int, default=None)
    args = parser.parse_args()

    config = RLVRConfig.from_yaml(args.reasoning)
    root = Path(config.output_dir)
    cycles = root / "cycles"
    state_path = root / "pipeline_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    start = args.start_iteration if args.start_iteration is not None else int(state.get("next_iteration", 0))
    checkpoint = str(state.get("checkpoint", args.checkpoint))
    reference = args.reference_checkpoint or args.checkpoint
    stop = config.iterations if args.stop_after is None else min(config.iterations, start + args.stop_after)

    for iteration in range(start, stop):
        cycle = cycles / f"iteration-{iteration:06d}"
        cycle.mkdir(parents=True, exist_ok=True)
        raw = cycle / "rollouts.raw.jsonl"
        scored = cycle / "rollouts.scored.jsonl"
        referenced = cycle / "rollouts.reference.jsonl"
        complete_marker = cycle / "update_complete.json"

        # A cycle is committed only after the resulting checkpoint path is atomically
        # recorded. If state.json was lost after a successful update, recover from it
        # instead of applying the same on-policy batch twice.
        if complete_marker.exists():
            committed = json.loads(complete_marker.read_text(encoding="utf-8"))
            checkpoint = str(committed["checkpoint"])
            atomic_state(state_path, {
                "next_iteration": iteration + 1,
                "checkpoint": checkpoint,
                "reference_checkpoint": reference,
                "model": args.model,
                "train": args.train,
                "reasoning": args.reasoning,
                "prompts": args.prompts,
            })
            continue

        if not raw.exists():
            run(
                [
                    sys.executable,
                    "scripts/generate_reasoning_rollouts.py",
                    "--model",
                    args.model,
                    "--checkpoint",
                    checkpoint,
                    "--reasoning",
                    args.reasoning,
                    "--prompts",
                    args.prompts,
                    "--output",
                    str(raw),
                    "--iteration",
                    str(iteration),
                ]
            )
        if not scored.exists():
            run(
                [
                    sys.executable,
                    "scripts/score_reasoning_rollouts.py",
                    "--input",
                    str(raw),
                    "--output",
                    str(scored),
                    "--reasoning",
                    args.reasoning,
                ]
            )
        update_input = scored
        if config.kl_beta > 0:
            if not referenced.exists():
                run(
                    [
                        sys.executable,
                        "scripts/score_rl_reference.py",
                        "--model",
                        args.model,
                        "--checkpoint",
                        reference,
                        "--tokenizer",
                        config.tokenizer_path,
                        "--input",
                        str(scored),
                        "--output",
                        str(referenced),
                        "--micro-batch-size",
                        str(config.micro_batch_size),
                    ]
                )
            update_input = referenced

        command = [
            sys.executable,
            "scripts/train_rlvr.py",
            "--model",
            args.model,
            "--train",
            args.train,
            "--reasoning",
            args.reasoning,
            "--checkpoint",
            checkpoint,
            "--rollouts",
            str(update_input),
            "--iteration",
            str(iteration),
        ]
        resolved_checkpoint = Path(checkpoint).expanduser().resolve()
        try:
            is_rl_checkpoint = resolved_checkpoint.is_relative_to(root.resolve())
        except AttributeError:  # Python 3.8 compatibility for downstream users
            is_rl_checkpoint = str(resolved_checkpoint).startswith(str(root.resolve()) + os.sep)
        if iteration > 0 or is_rl_checkpoint:
            command.append("--resume-optimizer")
        run(command)
        checkpoint = latest_checkpoint(root)
        atomic_state(complete_marker, {"iteration": iteration, "checkpoint": checkpoint})
        atomic_state(
            state_path,
            {
                "next_iteration": iteration + 1,
                "checkpoint": checkpoint,
                "reference_checkpoint": reference,
                "model": args.model,
                "train": args.train,
                "reasoning": args.reasoning,
                "prompts": args.prompts,
            },
        )
        if not config.save_rollouts:
            raw.unlink(missing_ok=True)
            scored.unlink(missing_ok=True)
            referenced.unlink(missing_ok=True)

    print(json.dumps({"next_iteration": stop, "checkpoint": checkpoint}, indent=2))


if __name__ == "__main__":
    main()
