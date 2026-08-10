from __future__ import annotations

from scripts.run_optimizer_matrix import median_present, nested_float


def test_nested_float_reads_phase_metric() -> None:
    payload = {"median_phase_timing": {"cuda_ms": {"optimizer": 12.5}}}

    assert nested_float(payload, "median_phase_timing", "cuda_ms", "optimizer") == 12.5
    assert nested_float(payload, "missing") is None


def test_median_present_ignores_missing_values() -> None:
    assert median_present([3.0, None, 1.0]) == 2.0
    assert median_present([None]) is None
