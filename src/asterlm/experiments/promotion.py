from __future__ import annotations

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
    "modal_cost_to_quality",
    "long_context_retrieval",
    "repeated_warm_throughput",
    "gpu_utilization_root_cause",
    "vram",
    "checkpoint_round_trip",
    "optimizer_scheduler_exact_resume",
    "data_cursor_exact_resume",
    "rng_state_resume",
    "interrupted_laptop_recovery",
    "modal_preemption_recovery",
    "local_to_modal_resume",
    "modal_to_local_resume",
    "modal_workspace_cross_resume",
    "huggingface_round_trip_hash",
    "wandb_history_resume",
    "laptop_inference",
    "modal_server_inference",
    "dense_baseline_regression",
    "correctness_and_data_quality_clear",
)


@dataclass(frozen=True, slots=True)
class PromotionGate:
    gate_id: str
    required: bool
    status: str
    evidence: tuple[str, ...]
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


def evaluate_promotion_gates(path: str | Path) -> PromotionDecision:
    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if payload.get("schema_version") != 1:
        raise ValueError("Promotion gates require schema_version: 1")
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
        evidence = tuple(str(item) for item in raw.get("evidence", []))
        if status == "passed" and not evidence:
            raise ValueError(f"Passed gate {gate_id!r} requires durable evidence")
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
    if observational != {"energy_and_power"}:
        raise ValueError(
            "energy_and_power must remain the sole observational, non-blocking telemetry gate"
        )
    blocking = tuple(gate.gate_id for gate in gates if gate.required and gate.status != "passed")
    return PromotionDecision(ready=not blocking, gates=tuple(gates), blocking_gate_ids=blocking)
