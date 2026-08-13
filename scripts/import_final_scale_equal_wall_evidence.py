#!/usr/bin/env python3
"""Promote final K3 scale only after two-seed equal-wall superiority."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from asterlm.artifacts import atomic_write_json, atomic_write_text, sha256_file

ROOT = Path(__file__).resolve().parents[1]
GATE_ID = "equal_wall_clock"
SMALL = "tier2-k3-latentmoe-868m-a483m:cutlass-grouped-k3-muon8bit-bf16"
LARGE = "tier2-k3-latentmoe-1p45b-a568m:cutlass-grouped-k3-muon8bit-bf16"


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _require_clean() -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError("Final scale evidence import requires a clean checkout")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def validate_analyses(analyses: list[dict[str, Any]]) -> dict[str, Any]:
    by_seed: dict[int, dict[str, Any]] = {}
    model_hashes: dict[str, set[str]] = {SMALL: set(), LARGE: set()}
    for analysis in analyses:
        if analysis.get("analysis_status") != "complete" or analysis.get("campaign_status") != "complete":
            raise ValueError("Scale campaign analysis is incomplete")
        candidates = analysis.get("candidates") or {}
        if set(candidates) != {SMALL, LARGE}:
            raise ValueError(f"Unexpected scale candidates: {sorted(candidates)}")
        runs = analysis.get("runs") or []
        seeds = {int(row["seed"]) for row in runs}
        if len(seeds) != 1 or len(runs) != 2:
            raise ValueError("Each scale artifact must contain one seed and two runs")
        seed = seeds.pop()
        if seed in by_seed:
            raise ValueError(f"Duplicate scale evidence for seed {seed}")
        if any(row.get("status") != "ok" or int(row.get("tokens_seen") or 0) != 1_048_576 for row in runs):
            raise ValueError(f"Seed {seed} contains an incomplete scale run")
        for row in runs:
            key = f"{row['candidate_id']}:{row['execution_variant']}"
            model_hashes[key].add(str(row.get("model_config_sha256") or ""))
        small = candidates[SMALL]
        large = candidates[LARGE]
        wall_improvement = float(small["equal_wall_loss_mean"]) - float(large["equal_wall_loss_mean"])
        token_improvement = float(small["final_eval_loss_mean"]) - float(large["final_eval_loss_mean"])
        flops_improvement = float(small["equal_active_flops_loss_mean"]) - float(
            large["equal_active_flops_loss_mean"]
        )
        if wall_improvement <= 0:
            raise ValueError(f"1.448B did not win equal-wall loss for seed {seed}")
        if token_improvement <= 0 or flops_improvement <= 0:
            raise ValueError(f"1.448B did not win token/FLOP controls for seed {seed}")
        by_seed[seed] = {
            "common_wall_seconds": float(analysis["common_budgets_by_seed"][str(seed)]["wall_clock_total_seconds"]),
            "small_equal_wall_loss": float(small["equal_wall_loss_mean"]),
            "large_equal_wall_loss": float(large["equal_wall_loss_mean"]),
            "large_equal_wall_improvement": wall_improvement,
            "large_equal_token_improvement": token_improvement,
            "large_equal_active_flops_improvement": flops_improvement,
            "small_median_tokens_per_second": float(small["median_training_tokens_per_second"]),
            "large_median_tokens_per_second": float(large["median_training_tokens_per_second"]),
        }
    if set(by_seed) != {1337, 2027}:
        raise ValueError(f"Expected seeds 1337 and 2027, got {sorted(by_seed)}")
    if any(len(values) != 1 or not next(iter(values)) for values in model_hashes.values()):
        raise ValueError("Materialized model configuration hashes differ across seeds")
    mean_wall_improvement = sum(row["large_equal_wall_improvement"] for row in by_seed.values()) / 2
    return {
        "seeds": by_seed,
        "mean_large_equal_wall_loss_improvement": mean_wall_improvement,
        "model_config_sha256": {key: next(iter(values)) for key, values in model_hashes.items()},
        "historical_control": (
            "The earlier two-seed 220M K3-vs-dense equal-wall failure remains recorded. "
            "This gate selects the final 1.448B deployable scale over its 868M fallback; "
            "dense mechanism quality remains separately covered by dense_baseline_regression."
        ),
    }


def _update_ledger(path: Path, proof: dict[str, str]) -> None:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for gate in payload.get("gates", []):
        if gate.get("id") == GATE_ID:
            gate["status"] = "passed"
            gate["evidence"] = [proof]
            gate.pop("note", None)
            atomic_write_text(path, yaml.safe_dump(payload, sort_keys=False))
            return
    raise RuntimeError(f"Promotion ledger is missing {GATE_ID}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", action="append", type=Path, required=True)
    parser.add_argument("--gates", type=Path, default=ROOT / "configs/experiments/promotion_gates.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/promotion-evidence")
    args = parser.parse_args()
    commit = _require_clean()
    assertions = validate_analyses([_load(path) for path in args.analysis])
    output = args.output / commit[:12] / "final-scale-equal-wall"
    output.mkdir(parents=True, exist_ok=True)
    sources = []
    for index, path in enumerate(args.analysis, start=1):
        target = output / f"quality-analysis-{index}.json"
        shutil.copyfile(path, target)
        sources.append({"path": target.relative_to(ROOT).as_posix(), "sha256": sha256_file(target)})
    result_path = output / "final-scale-equal-wall-result.json"
    atomic_write_json(
        result_path,
        {
            "schema_version": 1,
            "status": "passed",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "git_commit": commit,
            "source_artifacts": sources,
            "assertions": assertions,
        },
    )
    artifacts = [*sources, {"path": result_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(result_path)}]
    proof_path = output / f"{GATE_ID}-proof.json"
    atomic_write_json(
        proof_path,
        {
            "schema_version": 1,
            "gate_id": GATE_ID,
            "status": "passed",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "evaluator": {"name": "asterlm-final-scale-equal-wall", "version": "1", "git_commit": commit},
            "experiment_ids": ["k3-scale-quality-seed-1337", "k3-scale-quality-seed-2027"],
            "artifacts": artifacts,
        },
    )
    _update_ledger(
        args.gates,
        {"path": proof_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(proof_path)},
    )
    print(json.dumps(assertions, indent=2))


if __name__ == "__main__":
    main()
