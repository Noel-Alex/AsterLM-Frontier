from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMITTED_MATRIX = (
    ROOT / "docs/promotion-evidence/30e0cdd22875/native-8k-canary/matrix.json"
)


def _module():
    path = ROOT / "scripts/import_native_canary_evidence.py"
    spec = importlib.util.spec_from_file_location("import_native_canary_evidence", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_canary_importer_certifies_only_measured_system_claims() -> None:
    module = _module()
    assert set(module.CERTIFIED_GATES) == {
        "repeated_warm_throughput",
        "gpu_utilization_root_cause",
        "vram",
        "energy_and_power",
    }
    assert "equal_wall_clock" not in module.CERTIFIED_GATES
    assert "long_context_retrieval" not in module.CERTIFIED_GATES
    assert "laptop_inference" not in module.CERTIFIED_GATES


def test_native_canary_importer_accepts_all_three_source_pinned_trials() -> None:
    module = _module()
    assertions = module.validate_matrix(
        COMMITTED_MATRIX,
        module.load_json(COMMITTED_MATRIX),
    )
    assert len(assertions["trials"]) == 3
    assert assertions["aggregate"]["successful_repetitions"] == 3
    assert assertions["aggregate"]["median_gpu_utilization"] >= 90.0
    assert assertions["aggregate"]["median_peak_allocated_gib"] <= 11.25
    assert min(row["p10_gpu_utilization"] for row in assertions["trials"]) >= 85.0
