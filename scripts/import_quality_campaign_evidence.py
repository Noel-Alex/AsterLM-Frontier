#!/usr/bin/env python3
"""Import the completed matched-token architecture screen as durable evidence.

This script never runs training. It validates the complete source-pinned campaign,
copies its two summary artifacts into the tracked evidence archive, and certifies
only the claims directly supported by those artifacts. The equal-wall result is
recorded as a failure rather than being hidden or promoted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMPAIGN = ROOT / "runs/architecture-campaign/quality-mb4-rejection-16m8-two-seed-metrics-v2-20260811/quality-campaign.json"
DEFAULT_ANALYSIS = ROOT / "runs/architecture-campaign/quality-mb4-rejection-16m8-two-seed-metrics-v2-20260811/quality-analysis.json"
DEFAULT_GATES = ROOT / "configs/experiments/promotion_gates.yaml"
DEFAULT_OUTPUT = ROOT / "docs/promotion-evidence"

DENSE = "tier0-dense-mla-220m:torch-sdpa-bf16"
K3 = "tier2-k3-stable-latentmoe-220m:cutlass-grouped-bf16"
PASSED_GATES = ("equal_token_quality", "dense_baseline_regression")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    )
    value = result.stdout.strip().lower()
    if len(value) != 40:
        raise RuntimeError("A full Git commit is required")
    return value


def require_clean_checkout() -> None:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        raise RuntimeError("Quality evidence import must start from a clean checkout")


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def repo_relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def validate_campaign(campaign: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
    if campaign.get("schema_version") != 2 or campaign.get("status") != "complete":
        raise ValueError("Campaign is not a complete schema-v2 campaign")
    if analysis.get("schema_version") != 1:
        raise ValueError("Unexpected quality analysis schema")
    if analysis.get("campaign_status") != "complete" or analysis.get("analysis_status") != "complete":
        raise ValueError("Quality campaign or analysis is incomplete")
    if analysis.get("expected_runs") != 6 or analysis.get("complete_runs") != 6:
        raise ValueError("The matched campaign must contain all six runs")

    provenance = campaign.get("source_provenance") or {}
    execution = campaign.get("execution_checkout") or {}
    if provenance.get("dirty") is not False or execution.get("dirty") is not False:
        raise ValueError("Quality campaign was not source-pinned to a clean checkout")
    if provenance.get("git_commit") != execution.get("commit"):
        raise ValueError("Campaign source and execution commits differ")
    if campaign.get("seeds") != [1337, 2027] or campaign.get("tokens_per_candidate") != 16_777_216:
        raise ValueError("Unexpected seed or token budget")
    for seed in ("1337", "2027"):
        audit = (campaign.get("initialization_audits") or {}).get(seed) or {}
        k3_audit = (audit.get("candidates") or {}).get("tier2-k3-stable-latentmoe-220m") or {}
        if audit.get("status") != "ok" or k3_audit.get("mismatches") != []:
            raise ValueError(f"Shared initialization parity failed for seed {seed}")

    candidates = analysis.get("candidates") or {}
    dense = candidates.get(DENSE) or {}
    k3 = candidates.get(K3) or {}
    for name, row in ((DENSE, dense), (K3, k3)):
        if row.get("complete_seed_count") != 2 or row.get("expected_seed_count") != 2:
            raise ValueError(f"Candidate is missing replicated results: {name}")

    selected_runs = [
        row
        for row in analysis.get("runs", [])
        if row.get("candidate_id") in {
            "tier0-dense-mla-220m",
            "tier2-k3-stable-latentmoe-220m",
        }
    ]
    if len(selected_runs) != 4:
        raise ValueError("Expected two dense and two K3 source runs")
    for row in selected_runs:
        if row.get("status") != "ok" or row.get("tokens_seen") != 16_777_216:
            raise ValueError("Matched-token source run is incomplete")

    dense_loss = float(dense["final_eval_loss_mean"])
    k3_loss = float(k3["final_eval_loss_mean"])
    dense_wall = float(dense["equal_wall_loss_mean"])
    k3_wall = float(k3["equal_wall_loss_mean"])
    per_seed: dict[str, dict[str, dict[str, Any]]] = {}
    for row in selected_runs:
        per_seed.setdefault(str(row["seed"]), {})[str(row["candidate_id"])] = {
            "tokens_seen": row["tokens_seen"],
            "eval_main_loss": row["eval_main_loss"],
            "run_id": row["run_id"],
        }
    if not k3_loss <= dense_loss:
        raise ValueError("K3 did not meet the matched-token dense baseline")
    if not k3_wall > dense_wall:
        raise ValueError("Expected the recorded equal-wall K3 failure")

    return {
        "source_execution_commit": execution["commit"],
        "seeds": [1337, 2027],
        "tokens_per_candidate": 16_777_216,
        "complete_runs": 6,
        "matched_token": {
            "dense_final_eval_loss_mean": dense_loss,
            "k3_final_eval_loss_mean": k3_loss,
            "absolute_k3_improvement": dense_loss - k3_loss,
            "relative_k3_improvement_percent": (dense_loss - k3_loss) / dense_loss * 100.0,
            "passed": True,
        },
        "equal_wall": {
            "dense_loss_mean": dense_wall,
            "k3_loss_mean": k3_wall,
            "absolute_k3_regression": k3_wall - dense_wall,
            "passed": False,
        },
        "per_seed_runs": per_seed,
    }


def update_ledger(path: Path, proofs: dict[str, dict[str, str]], result: Path) -> None:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    seen: set[str] = set()
    for gate in payload.get("gates", []):
        gate_id = str(gate.get("id") or "")
        if gate_id in proofs:
            gate["status"] = "passed"
            gate["evidence"] = [proofs[gate_id]]
            seen.add(gate_id)
        elif gate_id == "equal_wall_clock":
            gate["status"] = "failed"
            gate["note"] = (
                "K3 regressed at the common wall-clock budget in the completed two-seed screen; "
                f"see {repo_relative(result)} ({sha256(result)}). Requires the final optimized native-context canary."
            )
    if seen != set(proofs):
        raise RuntimeError(f"Promotion ledger missing gates: {sorted(set(proofs) - seen)}")
    atomic_text(path, yaml.safe_dump(payload, sort_keys=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--gates", type=Path, default=DEFAULT_GATES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    require_clean_checkout()
    commit = git_commit()
    campaign = load_json(args.campaign)
    analysis = load_json(args.analysis)
    assertions = validate_campaign(campaign, analysis)

    output = args.output / commit[:12] / "quality-screen"
    output.mkdir(parents=True, exist_ok=True)
    campaign_copy = output / "quality-campaign.json"
    analysis_copy = output / "quality-analysis.json"
    shutil.copyfile(args.campaign, campaign_copy)
    shutil.copyfile(args.analysis, analysis_copy)
    result = output / "quality-screen-result.json"
    atomic_text(
        result,
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "imported_at_utc": datetime.now(timezone.utc).isoformat(),
                "importer_git_commit": commit,
                "source_artifacts": [
                    {"path": repo_relative(campaign_copy), "sha256": sha256(campaign_copy)},
                    {"path": repo_relative(analysis_copy), "sha256": sha256(analysis_copy)},
                ],
                "assertions": assertions,
                "certified_gates": list(PASSED_GATES),
                "failed_gates": ["equal_wall_clock"],
                "not_certified": [
                    "long_context_retrieval",
                    "repeated_warm_throughput",
                    "gpu_utilization_root_cause",
                    "vram",
                ],
            },
            indent=2,
        )
        + "\n",
    )

    artifacts = [
        {"path": repo_relative(campaign_copy), "sha256": sha256(campaign_copy)},
        {"path": repo_relative(analysis_copy), "sha256": sha256(analysis_copy)},
        {"path": repo_relative(result), "sha256": sha256(result)},
    ]
    records: dict[str, dict[str, str]] = {}
    for gate_id in PASSED_GATES:
        proof = output / f"{gate_id}-proof.json"
        atomic_text(
            proof,
            json.dumps(
                {
                    "schema_version": 1,
                    "gate_id": gate_id,
                    "status": "passed",
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "evaluator": {
                        "name": "asterlm-quality-campaign-importer",
                        "version": "1",
                        "git_commit": commit,
                    },
                    "experiment_ids": [
                        "architecture-screen-two-seed-16m8-20260811",
                        *[row["run_id"] for row in analysis["runs"]],
                    ],
                    "artifacts": artifacts,
                },
                indent=2,
            )
            + "\n",
        )
        records[gate_id] = {"path": repo_relative(proof), "sha256": sha256(proof)}
    update_ledger(args.gates, records, result)

    from asterlm.experiments import evaluate_promotion_gates

    decision = evaluate_promotion_gates(args.gates)
    print(json.dumps({"assertions": assertions, "promotion": decision.manifest()}, indent=2))


if __name__ == "__main__":
    main()
