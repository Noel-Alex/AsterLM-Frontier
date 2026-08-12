#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from asterlm.artifacts import atomic_write_json, atomic_write_text, sha256_file
from asterlm.experiments.long_context import (
    LONG_CONTEXT_CASE_SCHEMA_VERSION,
    build_retrieval_case,
    score_retrieval_case,
    summarize_retrieval_results,
)
from asterlm.generation import load_runtime
from asterlm.source_provenance import assert_current_checkout_source
from asterlm.training.checkpoint import resolve_checkpoint, verify_checkpoint


def _csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _load_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _append_result(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(result, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Token-exact, resumable long-context retrieval evaluation for base checkpoints"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--moe-implementation",
        choices=("reference", "grouped", "cutlass", "torch_grouped", "torchao_fp8"),
        default="reference",
        help="Physical MoE path, recorded separately from architecture semantics.",
    )
    parser.add_argument("--lengths", default="4096,8192,16384,32768,65536,131072")
    parser.add_argument("--depths", default="0.1,0.5,0.9")
    parser.add_argument("--tasks", default="exact_key,repeated_key,two_hop")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--prefill-chunk-size", type=int, default=2048)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--allow-dirty-source",
        action="store_true",
        help="Allow an exploratory run from modified source; never use it as promotion evidence.",
    )
    args = parser.parse_args()

    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    lengths = _csv_ints(args.lengths)
    depths = _csv_floats(args.depths)
    tasks = [item.strip() for item in args.tasks.split(",") if item.strip()]
    if not lengths or not depths or not tasks:
        raise ValueError("--lengths, --depths, and --tasks must not be empty")

    source = assert_current_checkout_source()
    if source is not None and source.get("dirty") and not args.allow_dirty_source:
        raise RuntimeError(
            "Long-context decision evidence requires a clean source checkout. Commit the evaluator "
            "or pass --allow-dirty-source for an explicitly non-promotable diagnostic."
        )
    resolved_checkpoint = resolve_checkpoint(args.checkpoint)
    if not resolved_checkpoint.is_dir():
        raise ValueError("Long-context decision evidence requires a manifested checkpoint directory")
    checkpoint_manifest = verify_checkpoint(resolved_checkpoint)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "cases.jsonl"
    summary_path = output / "summary.json"
    existing = _load_results(results_path)
    completed_ids = {str(row["case_id"]) for row in existing if row.get("status") == "ok"}

    tokenizer_path = Path(args.tokenizer).resolve()
    tokenizer_sha256 = sha256_file(tokenizer_path)
    model_artifact = next(
        artifact
        for artifact in checkpoint_manifest["artifacts"]
        if artifact["path"] == checkpoint_manifest["model_file"]
    )
    plan_identity = {
        "checkpoint_model_sha256": model_artifact["sha256"],
        "model": str(Path(args.model).resolve()) if args.model else None,
        "tokenizer_sha256": tokenizer_sha256,
        "device": args.device,
        "moe_implementation": args.moe_implementation,
        "lengths": lengths,
        "depths": depths,
        "tasks": tasks,
        "repeats": args.repeats,
        "seed": args.seed,
        "prefill_chunk_size": args.prefill_chunk_size,
        "source_commit": None if source is None else source.get("git_commit"),
        "source_tree_manifest_sha256": (
            None if source is None else source.get("tree_manifest_sha256")
        ),
    }
    plan_id = hashlib.sha256(
        json.dumps(plan_identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    plan = {
        "schema_version": LONG_CONTEXT_CASE_SCHEMA_VERSION,
        "plan_id": plan_id,
        "status": "running",
        "checkpoint": str(resolved_checkpoint.resolve()),
        "checkpoint_manifest": checkpoint_manifest,
        "model": str(Path(args.model).resolve()) if args.model else None,
        "tokenizer": str(tokenizer_path),
        "tokenizer_sha256": tokenizer_sha256,
        "device": args.device,
        "moe_implementation": args.moe_implementation,
        "lengths": lengths,
        "depths": depths,
        "tasks": tasks,
        "repeats": args.repeats,
        "seed": args.seed,
        "prefill_chunk_size": args.prefill_chunk_size,
        "source_provenance": source,
        "promotion_eligible_source": bool(source is None or not source.get("dirty")),
        "metric_semantics": {
            "quality": "teacher-forced answer-token NLL and greedy token recall",
            "length": "exact prompt plus answer tokens",
            "depth": "token offset, not character offset",
        },
    }
    existing_plan_path = output / "plan.json"
    if existing and existing_plan_path.is_file():
        existing_plan = json.loads(existing_plan_path.read_text(encoding="utf-8"))
        if existing_plan.get("plan_id") != plan_id:
            raise RuntimeError(
                "Existing long-context cases belong to a different immutable plan; "
                "preserve the output and choose a new --output directory."
            )
    atomic_write_json(output / "plan.json", plan)

    model, tokenizer = load_runtime(
        args.checkpoint,
        args.tokenizer,
        args.model,
        args.device,
        moe_implementation=args.moe_implementation,
    )

    results = list(existing)
    for task_index, task in enumerate(tasks):
        for length in lengths:
            for depth in depths:
                for repeat in range(args.repeats):
                    case_seed = (
                        args.seed
                        + task_index * 10_000_019
                        + length * 1_009
                        + round(depth * 10_000) * 97
                        + repeat
                    )
                    case = build_retrieval_case(
                        tokenizer,
                        task=task,
                        target_sequence_tokens=length,
                        depth=depth,
                        seed=case_seed,
                    )
                    if case.case_id in completed_ids:
                        continue
                    result = score_retrieval_case(
                        model,
                        case,
                        device=args.device,
                        prefill_chunk_size=args.prefill_chunk_size,
                    )
                    _append_result(results_path, result)
                    results.append(result)
                    completed_ids.add(case.case_id)
                    atomic_write_json(
                        summary_path,
                        {**plan, "status": "running", **summarize_retrieval_results(results)},
                    )

    atomic_write_json(
        summary_path,
        {**plan, "status": "complete", **summarize_retrieval_results(results)},
    )
    atomic_write_text(output / "COMPLETE", "complete\n")


if __name__ == "__main__":
    main()
