from __future__ import annotations

import json
from pathlib import Path

from asterlm.cloud import build_gcp_launch_plan, dispatch_gcp_launch_plan, load_gcp_profile

ROOT = Path(__file__).resolve().parents[1]


def _contract(tmp_path: Path) -> dict:
    path = tmp_path / "contract.json"
    payload = {
        "contract_id": "a" * 20,
        "provider": "gcp",
        "profile_alias": "google-credit",
        "estimated_spend_usd": 10.0,
        "blockers": ["provider_not_ready"],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_default_gcp_profile_is_safely_blocked_until_credentials_and_billing(tmp_path):
    profile = load_gcp_profile(ROOT / "configs/providers/gcp_boost.yaml", "google-credit")
    plan = build_gcp_launch_plan(_contract(tmp_path), profile, root=ROOT)
    assert plan["status"] == "blocked"
    assert "gcp_paid_billing_activation_required_for_gpu" in plan["blockers"]
    assert "gcp_gpu_quota_not_confirmed" in plan["blockers"]
    assert "gcp_dispatch_disabled" in plan["blockers"]
    serialized = json.dumps(plan)
    assert "HF_TOKEN" not in serialized
    assert "WANDB_API_KEY" not in serialized
    assert {row["accelerator"] for row in plan["attempts"]} == {
        "h100-80gb",
        "a100-80gb",
        "l4-24gb",
    }


def test_gcp_dispatch_is_dry_run_by_default(tmp_path):
    profile = load_gcp_profile(ROOT / "configs/providers/gcp_boost.yaml", "google-credit")
    plan = build_gcp_launch_plan(_contract(tmp_path), profile, root=ROOT)
    result = dispatch_gcp_launch_plan(plan)
    assert result["status"] == "dry_run"
