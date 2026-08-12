from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts" / "run_local_promotion_evidence.py"
    spec = importlib.util.spec_from_file_location("run_local_promotion_evidence", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_local_gate_map_cannot_certify_hardware_quality_or_provider_claims() -> None:
    module = _module()
    denied = {
        "equal_token_quality",
        "equal_wall_clock",
        "long_context_retrieval",
        "repeated_warm_throughput",
        "gpu_utilization_root_cause",
        "vram",
        "interrupted_laptop_recovery",
        "huggingface_round_trip_hash",
        "wandb_history_resume",
        "laptop_inference",
        "correctness_and_data_quality_clear",
    }
    assert denied.isdisjoint(module.GATE_ASSERTIONS)
    assert set(module.GATE_ASSERTIONS) == {
        "reference_correctness",
        "optimized_kernel_numerical_parity",
        "unit_and_integration_tests",
        "deterministic_reproducibility",
        "checkpoint_round_trip",
        "optimizer_scheduler_exact_resume",
        "data_cursor_exact_resume",
        "rng_state_resume",
    }


def test_local_evidence_refuses_a_dirty_checkout(monkeypatch) -> None:
    module = _module()

    class Result:
        stdout = " M src/asterlm/model.py\n"

    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: Result())
    try:
        module.require_clean_checkout()
    except RuntimeError as exc:
        assert "clean checkout" in str(exc)
        assert "src/asterlm/model.py" in str(exc)
    else:
        raise AssertionError("dirty checkout was accepted")
