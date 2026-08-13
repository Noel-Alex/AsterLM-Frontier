#!/usr/bin/env python3
"""Promote one completed, checkpoint-bound long-context retrieval evaluation."""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from asterlm.artifacts import atomic_write_json, atomic_write_text, sha256_file
from asterlm.training.checkpoint import verify_checkpoint

ROOT = Path(__file__).resolve().parents[1]
GATE_CONTRACTS = {
    "long_context_retrieval": {
        "run": "runs/aster-frontier-100b-stage1-4k",
        "minimum_tokens": 92_000_000_000,
        "lengths": {8192, 16384, 32768},
    },
    "stage2_long_context_retrieval": {
        "run": "runs/aster-frontier-100b-stage2-8k",
        "minimum_tokens": 3_000_000_000,
        "lengths": {16384, 32768, 65536},
    },
    "stage3_long_context_retrieval": {
        "run": "runs/aster-frontier-100b-stage3-16k",
        "minimum_tokens": 3_000_000_000,
        "lengths": {32768, 65536, 131072},
    },
}
REQUIRED_TASKS = {"exact_key", "repeated_key", "two_hop"}
REQUIRED_DEPTHS = {0.1, 0.5, 0.9}


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_clean_checkout() -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError("Long-context evidence import requires a clean checkout")


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def validate_summary(
    summary: dict[str, Any],
    gate_id: str,
    *,
    expected_checkpoint_root: Path,
    min_exact_accuracy: float,
) -> dict[str, Any]:
    contract = GATE_CONTRACTS[gate_id]
    if summary.get("schema_version") != 1 or summary.get("status") != "complete":
        raise ValueError("Retrieval summary is not complete schema-v1 evidence")
    if not summary.get("promotion_eligible_source"):
        raise ValueError("Retrieval evaluation used dirty or ineligible source")
    provenance = summary.get("source_provenance") or {}
    if provenance.get("dirty") is not False or len(str(provenance.get("git_commit") or "")) != 40:
        raise ValueError("Retrieval summary lacks clean full-commit provenance")
    checkpoint = Path(str(summary.get("checkpoint") or "")).resolve()
    try:
        checkpoint.relative_to(expected_checkpoint_root.resolve())
    except ValueError as exc:
        raise ValueError(
            f"{gate_id} must evaluate a checkpoint under {expected_checkpoint_root}"
        ) from exc
    manifest = summary.get("checkpoint_manifest") or {}
    if manifest.get("status") != "complete" or manifest.get("reason") != "complete":
        raise ValueError("Retrieval gate requires the completed stage checkpoint")
    if int(manifest.get("tokens_seen", 0)) < int(contract["minimum_tokens"]):
        raise ValueError("Retrieval checkpoint has not completed the declared stage token budget")
    lengths = {int(value) for value in summary.get("lengths", [])}
    if not set(contract["lengths"]).issubset(lengths):
        raise ValueError(f"Retrieval summary is missing required lengths: {contract['lengths']}")
    if not REQUIRED_TASKS.issubset(set(summary.get("tasks", []))):
        raise ValueError("Retrieval summary is missing a required task family")
    depths = {round(float(value), 4) for value in summary.get("depths", [])}
    if not REQUIRED_DEPTHS.issubset(depths) or int(summary.get("repeats", 0)) < 3:
        raise ValueError("Retrieval gate requires three depths and at least three repeats")
    accuracy = float(summary.get("exact_greedy_accuracy", -1.0))
    nll = float(summary.get("mean_answer_nll", float("nan")))
    if accuracy < min_exact_accuracy or not math.isfinite(nll):
        raise ValueError(
            f"Retrieval quality gate failed: exact={accuracy:.4f}, nll={nll}"
        )
    by_task = summary.get("by_task") or {}
    task_floor = min_exact_accuracy / 2.0
    for task in REQUIRED_TASKS:
        task_accuracy = float((by_task.get(task) or {}).get("exact_greedy_accuracy", -1.0))
        if task_accuracy < task_floor:
            raise ValueError(f"Retrieval task {task} is below its {task_floor:.3f} floor")
    return {
        "gate_id": gate_id,
        "checkpoint": str(checkpoint),
        "checkpoint_tokens": int(manifest["tokens_seen"]),
        "required_lengths": sorted(contract["lengths"]),
        "exact_greedy_accuracy": accuracy,
        "mean_answer_nll": nll,
        "minimum_exact_greedy_accuracy": min_exact_accuracy,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--gate", choices=sorted(GATE_CONTRACTS), required=True)
    parser.add_argument(
        "--gates", type=Path, default=ROOT / "configs/experiments/promotion_gates.yaml"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "docs/promotion-evidence"
    )
    parser.add_argument("--min-exact-accuracy", type=float, default=0.5)
    args = parser.parse_args()
    if not 0.0 <= args.min_exact_accuracy <= 1.0:
        raise ValueError("--min-exact-accuracy must be between zero and one")
    _require_clean_checkout()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    contract = GATE_CONTRACTS[args.gate]
    assertions = validate_summary(
        summary,
        args.gate,
        expected_checkpoint_root=ROOT / str(contract["run"]),
        min_exact_accuracy=args.min_exact_accuracy,
    )
    checkpoint = Path(assertions["checkpoint"])
    verify_checkpoint(checkpoint)
    cases = args.summary.parent / "cases.jsonl"
    complete = args.summary.parent / "COMPLETE"
    if not cases.is_file() or not complete.is_file():
        raise ValueError("Retrieval evidence is missing cases.jsonl or COMPLETE")

    commit = _git_commit()
    output = args.output / commit[:12] / args.gate
    output.mkdir(parents=True, exist_ok=True)
    summary_copy = output / "summary.json"
    cases_copy = output / "cases.jsonl"
    shutil.copyfile(args.summary, summary_copy)
    shutil.copyfile(cases, cases_copy)
    result = output / "result.json"
    atomic_write_json(
        result,
        {
            "schema_version": 1,
            "status": "complete",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "assertions": assertions,
        },
    )
    artifacts = [
        {"path": _relative(path), "sha256": sha256_file(path)}
        for path in (summary_copy, cases_copy, result)
    ]
    proof = output / f"{args.gate}-proof.json"
    atomic_write_json(
        proof,
        {
            "schema_version": 1,
            "gate_id": args.gate,
            "status": "passed",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "evaluator": {
                "name": "asterlm-long-context-importer",
                "version": "1",
                "git_commit": commit,
            },
            "experiment_ids": [str(summary["plan_id"])],
            "artifacts": artifacts,
        },
    )
    ledger = yaml.safe_load(args.gates.read_text(encoding="utf-8")) or {}
    matches = [gate for gate in ledger.get("gates", []) if gate.get("id") == args.gate]
    if len(matches) != 1:
        raise RuntimeError(f"Promotion ledger has {len(matches)} matches for {args.gate}")
    matches[0]["status"] = "passed"
    matches[0]["evidence"] = [{"path": _relative(proof), "sha256": sha256_file(proof)}]
    atomic_write_text(args.gates, yaml.safe_dump(ledger, sort_keys=False))
    print(json.dumps(assertions, indent=2))


if __name__ == "__main__":
    main()
