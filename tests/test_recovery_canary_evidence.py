from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMITTED_RESULT = (
    ROOT / "docs/promotion-evidence/5b3868a8979c/interrupted-recovery/result.json"
)


def _module():
    path = ROOT / "scripts/import_recovery_canary_evidence.py"
    spec = importlib.util.spec_from_file_location("import_recovery_canary_evidence", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_recovery_importer_accepts_the_real_signal_resume_result() -> None:
    module = _module()
    assertions = module.validate_result(module.load_json(COMMITTED_RESULT))
    assert assertions["signal"] == "SIGTERM"
    assert assertions["interrupted_returncode"] == 130
    assert assertions["final_step"] > assertions["stop_step"]
    assert assertions["final_tokens_seen"] > assertions["stop_tokens_seen"]
    assert assertions["resume_restored_data_cursor"] is True
    assert len(assertions["checkpoint_payloads_removed_after_audit"]) == 2
