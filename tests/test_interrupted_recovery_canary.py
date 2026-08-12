from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts/run_interrupted_recovery_canary.py"
    spec = importlib.util.spec_from_file_location("run_interrupted_recovery_canary", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_recovery_cleanup_is_bounded_to_promotion_canary(tmp_path, monkeypatch) -> None:
    module = _module()
    allowed = ROOT / "runs/promotion-canary/test-cleanup-bound"
    allowed.mkdir(parents=True, exist_ok=True)
    checkpoint = allowed / "checkpoint-00000001"
    checkpoint.mkdir()
    (checkpoint / "payload").write_text("x", encoding="utf-8")
    monkeypatch.setattr(module, "ROOT", ROOT)
    assert module.remove_checkpoint_payloads(allowed) == [checkpoint.name]
    assert not checkpoint.exists()

    outside = tmp_path / "checkpoint-run"
    outside.mkdir()
    try:
        module.remove_checkpoint_payloads(outside)
    except RuntimeError as exc:
        assert "Refusing cleanup" in str(exc)
    else:
        raise AssertionError("cleanup escaped the promotion-canary root")


def test_recovery_auditor_accepts_canonical_and_legacy_data_envelopes() -> None:
    module = _module()
    assert module.data_state_schema_version({"schema_version": 1}) == 1
    assert module.data_state_schema_version({"version": 1}) == 1
