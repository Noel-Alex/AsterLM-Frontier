from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMITTED_MATRIX = (
    ROOT / "docs/promotion-evidence/30e0cdd22875/native-8k-canary/matrix.json"
)
COMMITTED_RESULT = (
    ROOT
    / "docs/promotion-evidence/30e0cdd22875/native-8k-canary/native-8k-canary-result.json"
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


def test_committed_native_canary_evidence_contains_all_source_pinned_trials() -> None:
    # Raw profile outputs remain in the research archive. Clean-clone CI verifies
    # the compact, importer-produced proof and its committed source-matrix hash.
    result = json.loads(COMMITTED_RESULT.read_text(encoding="utf-8"))
    matrix_sha = hashlib.sha256(COMMITTED_MATRIX.read_bytes()).hexdigest()
    assert result["status"] == "passed"
    assert result["source_matrix"]["sha256"] == matrix_sha
    assertions = result["assertions"]
    assert len(assertions["trials"]) == 3
    assert assertions["aggregate"]["successful_repetitions"] == 3
    assert assertions["aggregate"]["median_gpu_utilization"] >= 90.0
    assert assertions["aggregate"]["median_peak_allocated_gib"] <= 11.25
    assert min(row["p10_gpu_utilization"] for row in assertions["trials"]) >= 85.0
