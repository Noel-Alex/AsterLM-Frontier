#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


DEFAULT_MODELS = [
    "configs/model/aster_dense_challenger_666m.yaml",
    "configs/model/aster_moe_frontier_893m_a484m.yaml",
    "configs/model/aster_moe_target_1p51b_a623m.yaml",
    "configs/model/aster_moe_frontier_893m_loqt.yaml",
]


def latest_eval(metrics: Path) -> dict:
    best: dict = {}
    if not metrics.exists():
        return best
    for line in metrics.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if "eval_main_loss" in row:
            best = row
    return best


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train architecture/precision candidates for equal tokens and rank by held-out loss"
    )
    parser.add_argument("--data", default="configs/data/pretrain_frontier_clean.yaml")
    parser.add_argument("--train", default="configs/train/probe_quality_2k.yaml")
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--tokens", type=int, default=100_000_000)
    parser.add_argument("--output", default="runs/quality-ablations")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    models = args.model or DEFAULT_MODELS
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load(Path(args.train).read_text(encoding="utf-8"))
    records: list[dict] = []

    for model in models:
        name = Path(model).stem
        run_dir = root / name
        config = json.loads(json.dumps(base))
        config["train"]["output_dir"] = str(run_dir)
        config["train"]["max_tokens"] = args.tokens
        train_path = root / f"{name}-train.yaml"
        train_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        command = [
            sys.executable,
            "scripts/train_pretrain.py",
            "--model",
            model,
            "--train",
            str(train_path),
            "--data",
            args.data,
        ]
        print("$", " ".join(command), flush=True)
        completed = subprocess.run(command, check=False)
        evaluation = latest_eval(run_dir / "metrics.jsonl")
        record = {
            "model": model,
            "returncode": completed.returncode,
            "eval_main_loss": evaluation.get("eval_main_loss"),
            "eval_perplexity": evaluation.get("eval_perplexity"),
            "tokens_seen": evaluation.get("tokens_seen"),
            "run_dir": str(run_dir),
        }
        records.append(record)
        (root / "results.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        if completed.returncode and not args.continue_on_error:
            raise SystemExit(completed.returncode)

    ranked = sorted(
        records,
        key=lambda row: (
            row["eval_main_loss"] is None,
            row["eval_main_loss"] if row["eval_main_loss"] is not None else 1e9,
        ),
    )
    (root / "ranked.json").write_text(json.dumps(ranked, indent=2), encoding="utf-8")
    print(json.dumps(ranked, indent=2))


if __name__ == "__main__":
    main()
