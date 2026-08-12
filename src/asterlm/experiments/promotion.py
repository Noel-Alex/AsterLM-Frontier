from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REQUIRED_FINAL_RUN_GATES = (
    "reference_correctness",
    "optimized_kernel_numerical_parity",
    "unit_and_integration_tests",
    "deterministic_reproducibility",
    "equal_token_quality",
    "equal_wall_clock",
    "long_context_retrieval",
    "repeated_warm_throughput",
    "gpu_utilization_root_cause",
    "vram",
    "checkpoint_round_trip",
    "optimizer_scheduler_exact_resume",
    "data_cursor_exact_resume",
    "rng_state_resume",
    "interrupted_laptop_recovery",
    "huggingface_round_trip_hash",
    "wandb_history_resume",
    "laptop_inference",
    "dense_baseline_regression",
    "correctness_and_data_quality_clear",
)

MODAL_PROMOTION_GATES = (
    "modal_cost_to_quality",
    "modal_preemption_recovery",
    "local_to_modal_resume",
    "modal_to_local_resume",
    "modal_workspace_cross_resume",
    "modal_server_inference",
)

OBSERVATIONAL_GATES = ("energy_and_power", *MODAL_PROMOTION_GATES)


@dataclass(frozen=True, slots=True)
class PromotionEvidence:
    path: str
    sha256: str
    evaluator: str
    evaluator_version: str
    git_commit: str
    experiment_ids: tuple[str, ...]
    artifact_count: int


@dataclass(frozen=True, slots=True)
class PromotionGate:
    gate_id: str
    required: bool
    status: str
    evidence: tuple[PromotionEvidence, ...]
    note: str | None


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    ready: bool
    gates: tuple[PromotionGate, ...]
    blocking_gate_ids: tuple[str, ...]

    def manifest(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "passed": sum(gate.status == "passed" for gate in self.gates),
            "total_required": sum(gate.required for gate in self.gates),
            "blocking_gate_ids": list(self.blocking_gate_ids),
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(root: Path, value: str, *, label: str) -> Path:
    candidate = Path(value)
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes promotion repo_root: {value}") from exc
    return resolved


def _verified_evidence(
    gate_id: str, raw: Any, *, repo_root: Path
) -> PromotionEvidence:
    if not isinstance(raw, dict):
        raise TypeError(
            f"Passed gate {gate_id!r} evidence must be a structured path/sha256 record"
        )
    path_value = str(raw.get("path") or "")
    expected_hash = str(raw.get("sha256") or "").lower()
    if not path_value or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise ValueError(f"Passed gate {gate_id!r} evidence requires path and SHA-256")
    proof_path = _inside(repo_root, path_value, label=f"Gate {gate_id!r} evidence")
    if not proof_path.is_file():
        raise ValueError(f"Passed gate {gate_id!r} evidence is missing: {proof_path}")
    observed_hash = _sha256(proof_path)
    if observed_hash != expected_hash:
        raise ValueError(
            f"Passed gate {gate_id!r} evidence hash mismatch: "
            f"expected {expected_hash}, observed {observed_hash}"
        )
    try:
        proof = json.loads(proof_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Gate {gate_id!r} evidence is not valid JSON: {proof_path}") from exc
    if not isinstance(proof, dict) or proof.get("schema_version") != 1:
        raise ValueError(f"Gate {gate_id!r} evidence requires proof schema_version 1")
    if proof.get("gate_id") != gate_id or proof.get("status") != "passed":
        raise ValueError(f"Gate {gate_id!r} evidence does not attest that exact gate passed")
    evaluator = proof.get("evaluator")
    if not isinstance(evaluator, dict):
        raise TypeError(f"Gate {gate_id!r} evidence requires evaluator provenance")
    evaluator_name = str(evaluator.get("name") or "")
    evaluator_version = str(evaluator.get("version") or "")
    git_commit = str(evaluator.get("git_commit") or "").lower()
    if not evaluator_name or not evaluator_version or not re.fullmatch(
        r"[0-9a-f]{40,64}", git_commit
    ):
        raise ValueError(
            f"Gate {gate_id!r} evaluator requires name, version, and full Git commit"
        )
    experiment_ids = proof.get("experiment_ids")
    if not isinstance(experiment_ids, list) or not experiment_ids or not all(
        isinstance(item, str) and item.strip() for item in experiment_ids
    ):
        raise ValueError(f"Gate {gate_id!r} evidence requires experiment_ids")
    artifacts = proof.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError(f"Gate {gate_id!r} evidence requires hashed source artifacts")
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict):
            raise TypeError(f"Gate {gate_id!r} artifact {index} must be structured")
        artifact_path = _inside(
            repo_root,
            str(artifact.get("path") or ""),
            label=f"Gate {gate_id!r} artifact {index}",
        )
        artifact_hash = str(artifact.get("sha256") or "").lower()
        if not artifact_path.is_file() or not re.fullmatch(r"[0-9a-f]{64}", artifact_hash):
            raise ValueError(f"Gate {gate_id!r} artifact {index} is missing or unpinned")
        if _sha256(artifact_path) != artifact_hash:
            raise ValueError(f"Gate {gate_id!r} artifact {index} hash mismatch")
    return PromotionEvidence(
        path=path_value,
        sha256=expected_hash,
        evaluator=evaluator_name,
        evaluator_version=evaluator_version,
        git_commit=git_commit,
        experiment_ids=tuple(experiment_ids),
        artifact_count=len(artifacts),
    )


def evaluate_promotion_gates(path: str | Path) -> PromotionDecision:
    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if payload.get("schema_version") != 2:
        raise ValueError("Promotion gates require schema_version: 2")
    repo_root_value = str(payload.get("repo_root") or "../..")
    repo_root = (source.resolve().parent / repo_root_value).resolve()
    if not repo_root.is_dir():
        raise ValueError(f"Promotion repo_root does not exist: {repo_root}")
    statuses = set(payload.get("allowed_statuses", []))
    if statuses != {"not_run", "running", "passed", "failed", "blocked"}:
        raise ValueError("Promotion gate allowed_statuses changed unexpectedly")

    gates: list[PromotionGate] = []
    seen: set[str] = set()
    for raw in payload.get("gates", []):
        gate_id = str(raw.get("id", ""))
        if gate_id in seen:
            raise ValueError(f"Duplicate promotion gate: {gate_id}")
        seen.add(gate_id)
        status = str(raw.get("status", ""))
        if status not in statuses:
            raise ValueError(f"Gate {gate_id!r} has invalid status {status!r}")
        raw_evidence = raw.get("evidence", [])
        if not isinstance(raw_evidence, list):
            raise TypeError(f"Gate {gate_id!r} evidence must be a list")
        if status == "passed" and not raw_evidence:
            raise ValueError(f"Passed gate {gate_id!r} requires durable evidence")
        evidence = (
            tuple(
                _verified_evidence(gate_id, item, repo_root=repo_root)
                for item in raw_evidence
            )
            if status == "passed"
            else ()
        )
        gates.append(
            PromotionGate(
                gate_id=gate_id,
                required=bool(raw.get("required", True)),
                status=status,
                evidence=evidence,
                note=str(raw["note"]) if raw.get("note") is not None else None,
            )
        )

    expected = set(REQUIRED_FINAL_RUN_GATES)
    actual_required = {gate.gate_id for gate in gates if gate.required}
    if actual_required != expected:
        missing = sorted(expected - actual_required)
        unexpected = sorted(actual_required - expected)
        raise ValueError(f"Final-run gate set changed; missing={missing}, unexpected={unexpected}")
    observational = {gate.gate_id for gate in gates if not gate.required}
    if observational != set(OBSERVATIONAL_GATES):
        raise ValueError(
            "Observational/provider gate set changed unexpectedly"
        )
    blocking = tuple(gate.gate_id for gate in gates if gate.required and gate.status != "passed")
    return PromotionDecision(ready=not blocking, gates=tuple(gates), blocking_gate_ids=blocking)
