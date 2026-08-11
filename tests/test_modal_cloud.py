from __future__ import annotations

import json
from pathlib import Path

import yaml

from asterlm.cloud import build_modal_launch_plan, dispatch_modal_launch_plan, load_modal_profile

ROOT = Path(__file__).resolve().parents[1]


def _contract(tmp_path: Path, **overrides) -> tuple[dict, Path]:
    payload = {
        "contract_id": "a" * 20,
        "provider": "modal",
        "profile_alias": "noelalex404",
        "git_commit": "b" * 40,
        "timeout_minutes": 180,
        "estimated_spend_usd": 10.0,
        "blockers": ["provider_not_ready"],
    }
    payload.update(overrides)
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload, path


def test_default_modal_profile_is_safely_blocked_and_secret_free(tmp_path):
    contract, contract_path = _contract(tmp_path)
    profile = load_modal_profile(ROOT / "configs/providers/modal_boost.yaml", "noelalex404")
    plan = build_modal_launch_plan(
        contract, profile, root=ROOT, contract_path=contract_path
    )
    assert plan["status"] == "blocked"
    assert "modal_base_image_not_configured" in plan["blockers"]
    assert "modal_dispatch_disabled" in plan["blockers"]
    assert plan["volume_version"] == 2
    assert plan["volumes"]["/opt/aster/data"] == "aster-data-noelalex404"
    serialized = json.dumps(plan)
    assert "HF_TOKEN" not in serialized
    assert "WANDB_API_KEY" not in serialized
    assert [row["gpu"] for row in plan["attempts"]][:3] == ["B300", "H200", "H100!"]


def test_modal_profiles_do_not_share_persistent_volumes():
    profiles = [
        load_modal_profile(ROOT / "configs/providers/modal_boost.yaml", alias)
        for alias in ("noelalex404", "friend-1", "friend-2")
    ]
    for field in ("dataset_volume", "cache_volume", "checkpoint_volume"):
        assert len({getattr(profile, field) for profile in profiles}) == len(profiles)


def test_modal_rejects_unpinned_image_and_incompatible_candidate(tmp_path):
    config = yaml.safe_load((ROOT / "configs/providers/modal_boost.yaml").read_text(encoding="utf-8"))
    raw = config["profiles"]["noelalex404"]
    raw["base_image"] = "nvcr.io/nvidia/pytorch:latest"
    raw["cuda_minor"] = "13.0"
    config_path = tmp_path / "modal.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    contract, contract_path = _contract(tmp_path, blockers=[])
    profile = load_modal_profile(config_path, "noelalex404")
    plan = build_modal_launch_plan(
        contract, profile, root=ROOT, contract_path=contract_path
    )
    assert "modal_base_image_not_digest_pinned" in plan["blockers"]
    assert plan["attempts"][0]["blockers"] == ["requires_cuda_13.1_or_newer"]


def test_modal_dispatch_is_dry_run_by_default(tmp_path):
    contract, contract_path = _contract(tmp_path)
    profile = load_modal_profile(ROOT / "configs/providers/modal_boost.yaml", "noelalex404")
    plan = build_modal_launch_plan(
        contract, profile, root=ROOT, contract_path=contract_path
    )
    result = dispatch_modal_launch_plan(plan)
    assert result["status"] == "dry_run"
