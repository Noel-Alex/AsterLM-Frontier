#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GATES = ROOT / "configs" / "experiments" / "promotion_gates.yaml"
DEFAULT_OUTPUT = ROOT / "docs" / "promotion-evidence"

# A gate may reuse a test only when that exact test directly asserts the named
# property. Hardware/quality/provider gates are intentionally absent; this tool
# cannot turn a generic green suite into evidence for those claims.
GATE_ASSERTIONS: dict[str, tuple[str, ...]] = {
    "reference_correctness": (
        "tests.test_model.test_cached_decode_matches_full_forward",
        "tests.test_model.test_chunked_prefill_matches_full_forward_last_token",
    ),
    "optimized_kernel_numerical_parity": (
        "tests.test_moe_cutlass_cuda.test_grouped_moe_matches_dropless_reference[cutlass]",
    ),
    "unit_and_integration_tests": ("__entire_suite__",),
    "deterministic_reproducibility": (
        "tests.test_quality_campaign.test_named_initialization_audit_accepts_shared_projection_parity",
        "tests.test_exact_training_resume.test_uninterrupted_and_reconstructed_training_are_exact",
    ),
    "checkpoint_round_trip": (
        "tests.test_checkpoint.test_checkpoint_round_trip",
        "tests.test_checkpoint.test_checkpoint_hash_detects_corruption",
    ),
    "optimizer_scheduler_exact_resume": (
        "tests.test_exact_training_resume.test_uninterrupted_and_reconstructed_training_are_exact",
    ),
    "data_cursor_exact_resume": (
        "tests.test_data.test_packed_data_cursor_resumes_exactly_without_replaying_prior_batches",
        "tests.test_exact_training_resume.test_uninterrupted_and_reconstructed_training_are_exact",
    ),
    "rng_state_resume": (
        "tests.test_exact_training_resume.test_uninterrupted_and_reconstructed_training_are_exact",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip().lower()
    if len(value) != 40:
        raise RuntimeError("Promotion evidence requires a full Git commit")
    return value


def require_clean_checkout() -> None:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    dirty = result.stdout.strip()
    if dirty:
        raise RuntimeError(
            "Promotion evidence must start from a clean checkout so its evaluator "
            f"commit identifies the exact source under test. Dirty paths:\n{dirty}"
        )


def junit_cases(path: Path) -> tuple[dict[str, str], dict[str, int]]:
    root = ElementTree.parse(path).getroot()
    cases: dict[str, str] = {}
    for case in root.iter("testcase"):
        classname = str(case.attrib.get("classname") or "")
        name = str(case.attrib.get("name") or "")
        identity = f"{classname}.{name}"
        if case.find("failure") is not None:
            state = "failed"
        elif case.find("error") is not None:
            state = "error"
        elif case.find("skipped") is not None:
            state = "skipped"
        else:
            state = "passed"
        cases[identity] = state
    totals = {
        "tests": len(cases),
        "passed": sum(state == "passed" for state in cases.values()),
        "failed": sum(state == "failed" for state in cases.values()),
        "errors": sum(state == "error" for state in cases.values()),
        "skipped": sum(state == "skipped" for state in cases.values()),
    }
    return cases, totals


def repo_relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def record_proof(
    *,
    gate_id: str,
    commit: str,
    output: Path,
    junit: Path,
    result: Path,
    asserted_cases: tuple[str, ...],
) -> tuple[Path, dict[str, str]]:
    artifacts = []
    for path in (junit, result):
        artifacts.append({"path": repo_relative(path), "sha256": sha256(path)})
    proof = {
        "schema_version": 1,
        "gate_id": gate_id,
        "status": "passed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluator": {
            "name": "asterlm-local-promotion-evaluator",
            "version": "1",
            "git_commit": commit,
        },
        "experiment_ids": [f"local-suite-{commit[:12]}", *asserted_cases],
        "asserted_test_cases": list(asserted_cases),
        "artifacts": artifacts,
    }
    proof_path = output / f"{gate_id}-proof.json"
    atomic_text(proof_path, json.dumps(proof, indent=2) + "\n")
    return proof_path, {"path": repo_relative(proof_path), "sha256": sha256(proof_path)}


def update_gate_ledger(
    path: Path,
    evidence: dict[str, dict[str, str]],
) -> None:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    remaining = set(evidence)
    for gate in payload.get("gates", []):
        gate_id = str(gate.get("id") or "")
        if gate_id not in evidence:
            continue
        gate["status"] = "passed"
        existing = gate.get("evidence") if isinstance(gate.get("evidence"), list) else []
        gate["evidence"] = [
            item for item in existing if item.get("path") != evidence[gate_id]["path"]
        ] + [evidence[gate_id]]
        remaining.remove(gate_id)
    if remaining:
        raise RuntimeError(f"Gate ledger is missing: {', '.join(sorted(remaining))}")
    atomic_text(path, yaml.safe_dump(payload, sort_keys=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the complete local suite and record only directly asserted promotion gates"
    )
    parser.add_argument("--gates", type=Path, default=DEFAULT_GATES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    require_clean_checkout()
    commit = git_commit()
    output = args.output / commit[:12]
    output.mkdir(parents=True, exist_ok=True)
    junit = output / "pytest-full-suite.junit.xml"
    command = [sys.executable, "-m", "pytest", "-q", f"--junitxml={junit}"]
    completed = subprocess.run(command, cwd=ROOT, text=True)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)

    cases, totals = junit_cases(junit)
    if totals["failed"] or totals["errors"] or totals["skipped"]:
        raise RuntimeError(f"Promotion suite was not all-pass: {totals}")
    gate_results: dict[str, dict[str, object]] = {}
    certified: dict[str, tuple[str, ...]] = {}
    for gate_id, assertions in GATE_ASSERTIONS.items():
        if assertions == ("__entire_suite__",):
            passed = totals["tests"] > 0 and totals["passed"] == totals["tests"]
            observed = (f"entire_suite:{totals['tests']}",)
        else:
            missing = [identity for identity in assertions if cases.get(identity) != "passed"]
            passed = not missing
            observed = assertions
        gate_results[gate_id] = {
            "passed": passed,
            "assertions": list(observed),
        }
        if passed:
            certified[gate_id] = observed

    expected = set(GATE_ASSERTIONS)
    if set(certified) != expected:
        failed = sorted(expected - set(certified))
        raise RuntimeError(f"Required local evidence assertions did not pass: {failed}")

    result = output / "local-suite-result.json"
    atomic_text(
        result,
        json.dumps(
            {
                "schema_version": 1,
                "status": "passed",
                "git_commit": commit,
                "command": command,
                "junit_sha256": sha256(junit),
                "totals": totals,
                "gate_results": gate_results,
                "not_certified_by_design": [
                    "quality comparisons",
                    "wall-clock comparisons",
                    "GPU utilization or VRAM",
                    "interruption signal recovery",
                    "Hub or W&B network round trips",
                    "inference quality/performance",
                    "clean-corpus quality",
                    "provider-specific recovery",
                ],
            },
            indent=2,
        )
        + "\n",
    )

    records: dict[str, dict[str, str]] = {}
    for gate_id, assertions in certified.items():
        _, records[gate_id] = record_proof(
            gate_id=gate_id,
            commit=commit,
            output=output,
            junit=junit,
            result=result,
            asserted_cases=assertions,
        )
    update_gate_ledger(args.gates, records)

    from asterlm.experiments import evaluate_promotion_gates

    decision = evaluate_promotion_gates(args.gates)
    print(
        json.dumps(
            {
                "certified": sorted(certified),
                "suite": totals,
                "promotion": decision.manifest(),
                "output": repo_relative(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
