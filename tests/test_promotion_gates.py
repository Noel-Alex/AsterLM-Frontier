from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from asterlm.experiments import MODAL_PROMOTION_GATES, REQUIRED_FINAL_RUN_GATES, evaluate_promotion_gates

ROOT = Path(__file__).resolve().parents[1]
GATES = ROOT / "configs/experiments/promotion_gates.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_evidence(root: Path, gate_id: str) -> dict[str, str]:
    artifact = root / f"{gate_id}-result.json"
    artifact.write_text(json.dumps({"gate_id": gate_id, "metric": 1.0}), encoding="utf-8")
    proof = root / f"{gate_id}-proof.json"
    proof.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "gate_id": gate_id,
                "status": "passed",
                "evaluator": {
                    "name": "test-evaluator",
                    "version": "1",
                    "git_commit": "a" * 40,
                },
                "experiment_ids": [f"experiment-{gate_id}"],
                "artifacts": [{"path": artifact.name, "sha256": _sha256(artifact)}],
            }
        ),
        encoding="utf-8",
    )
    return {"path": proof.name, "sha256": _sha256(proof)}


def test_project_final_run_is_locked_by_unpassed_required_gates_with_energy_observational_only():
    decision = evaluate_promotion_gates(GATES)
    assert len(decision.gates) == 29
    expected_blocking = tuple(
        gate.gate_id
        for gate in decision.gates
        if gate.gate_id in REQUIRED_FINAL_RUN_GATES and gate.status != "passed"
    )
    assert decision.blocking_gate_ids == expected_blocking
    assert decision.ready == (not expected_blocking)
    energy = next(gate for gate in decision.gates if gate.gate_id == "energy_and_power")
    assert not energy.required
    assert all(
        not next(gate for gate in decision.gates if gate.gate_id == gate_id).required
        for gate_id in MODAL_PROMOTION_GATES
    )


def test_stage1_is_not_circularly_blocked_by_post_stage1_context_evidence():
    stage1 = evaluate_promotion_gates(GATES, phase="stage1")
    stage2 = evaluate_promotion_gates(GATES, phase="stage2")
    stage3 = evaluate_promotion_gates(GATES, phase="stage3")
    stage4 = evaluate_promotion_gates(GATES, phase="stage4")
    assert "long_context_retrieval" not in stage1.required_gate_ids
    assert "long_context_retrieval" not in stage1.blocking_gate_ids
    assert "long_context_retrieval" in stage2.required_gate_ids
    assert "long_context_retrieval" in stage2.blocking_gate_ids
    assert "stage2_long_context_retrieval" not in stage2.required_gate_ids
    assert "stage2_long_context_retrieval" in stage3.required_gate_ids
    assert "stage3_long_context_retrieval" not in stage3.required_gate_ids
    assert "stage3_long_context_retrieval" in stage4.required_gate_ids


def test_passed_gate_requires_evidence(tmp_path):
    payload = yaml.safe_load(GATES.read_text(encoding="utf-8"))
    payload["repo_root"] = "."
    payload["gates"][0]["status"] = "passed"
    payload["gates"][0]["evidence"] = []
    path = tmp_path / "gates.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="durable evidence"):
        evaluate_promotion_gates(path)


def test_all_gates_pass_only_with_evidence(tmp_path):
    payload = yaml.safe_load(GATES.read_text(encoding="utf-8"))
    payload["repo_root"] = "."
    for gate in payload["gates"]:
        gate["status"] = "passed"
        gate["evidence"] = [_write_evidence(tmp_path, gate["id"])]
    path = tmp_path / "gates.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    decision = evaluate_promotion_gates(path)
    assert decision.ready
    assert not decision.blocking_gate_ids


def test_passed_gate_rejects_tampered_evidence(tmp_path):
    payload = yaml.safe_load(GATES.read_text(encoding="utf-8"))
    payload["repo_root"] = "."
    gate = payload["gates"][0]
    gate["status"] = "passed"
    gate["evidence"] = [_write_evidence(tmp_path, gate["id"])]
    proof = tmp_path / gate["evidence"][0]["path"]
    proof.write_text("{}", encoding="utf-8")
    path = tmp_path / "gates.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="evidence hash mismatch"):
        evaluate_promotion_gates(path)
