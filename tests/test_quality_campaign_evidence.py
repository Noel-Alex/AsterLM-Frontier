from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts/import_quality_campaign_evidence.py"
    spec = importlib.util.spec_from_file_location("import_quality_campaign_evidence", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_quality_importer_certifies_only_matched_token_claims() -> None:
    module = _module()
    assert module.PASSED_GATES == ("equal_token_quality", "dense_baseline_regression")
    assert "equal_wall_clock" not in module.PASSED_GATES
    assert "gpu_utilization_root_cause" not in module.PASSED_GATES
    assert "long_context_retrieval" not in module.PASSED_GATES


def test_quality_importer_accepts_the_complete_source_pinned_campaign() -> None:
    module = _module()
    assertions = module.validate_campaign(
        module.load_json(module.DEFAULT_CAMPAIGN),
        module.load_json(module.DEFAULT_ANALYSIS),
    )
    assert assertions["complete_runs"] == 6
    assert assertions["matched_token"]["passed"] is True
    assert assertions["matched_token"]["k3_final_eval_loss_mean"] < assertions["matched_token"]["dense_final_eval_loss_mean"]
    assert assertions["equal_wall"]["passed"] is False
    assert assertions["equal_wall"]["k3_loss_mean"] > assertions["equal_wall"]["dense_loss_mean"]
