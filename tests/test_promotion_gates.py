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


def test_project_final_run_is_locked_by_26_gates_with_energy_observational_only():
    decision = evaluate_promotion_gates(GATES)
    assert not decision.ready
    assert len(decision.gates) == 27
    assert decision.blocking_gate_ids == REQUIRED_FINAL_RUN_GATES
    energy = next(gate for gate in decision.gates if gate.gate_id == "energy_and_power")
    assert not energy.required
    assert all(
        not next(gate for gate in decision.gates if gate.gate_id == gate_id).required
        for gate_id in MODAL_PROMOTION_GATES
    )


def test_passed_gate_requires_evidence(tmp_path):
    payload = yaml.safe_load(GATES.read_text(encoding="utf-8"))
    payload["repo_root"] = "."
    payload["gates"][0]["status"] = "passed"
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
