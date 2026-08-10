from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from asterlm.experiments import REQUIRED_FINAL_RUN_GATES, evaluate_promotion_gates

ROOT = Path(__file__).resolve().parents[1]
GATES = ROOT / "configs/experiments/promotion_gates.yaml"


def test_project_final_run_is_locked_by_26_gates_with_energy_observational_only():
    decision = evaluate_promotion_gates(GATES)
    assert not decision.ready
    assert len(decision.gates) == 27
    assert decision.blocking_gate_ids == REQUIRED_FINAL_RUN_GATES
    energy = next(gate for gate in decision.gates if gate.gate_id == "energy_and_power")
    assert not energy.required


def test_passed_gate_requires_evidence(tmp_path):
    payload = yaml.safe_load(GATES.read_text(encoding="utf-8"))
    payload["gates"][0]["status"] = "passed"
    path = tmp_path / "gates.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="durable evidence"):
        evaluate_promotion_gates(path)


def test_all_gates_pass_only_with_evidence(tmp_path):
    payload = yaml.safe_load(GATES.read_text(encoding="utf-8"))
    for gate in payload["gates"]:
        gate["status"] = "passed"
        gate["evidence"] = [f"runs/evidence/{gate['id']}.json"]
    path = tmp_path / "gates.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    decision = evaluate_promotion_gates(path)
    assert decision.ready
    assert not decision.blocking_gate_ids
