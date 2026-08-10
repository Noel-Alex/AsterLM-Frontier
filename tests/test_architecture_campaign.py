from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from asterlm.config import AsterConfig
from asterlm.experiments import load_architecture_campaign, materialize_architecture_campaign

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "configs/experiments/architecture_campaign.yaml"


def test_project_architecture_campaign_is_valid_and_materializable(tmp_path):
    campaign = load_architecture_campaign(CAMPAIGN, repo_root=ROOT)
    assert campaign.context_lengths == (4096, 8192, 16384, 32768, 65536, 131072)
    assert campaign.candidates[0].candidate_id == "tier0-dense-mla-480m"
    manifest = materialize_architecture_campaign(campaign, tmp_path)
    assert len(manifest["candidates"]) == len(campaign.candidates)
    for item in manifest["candidates"].values():
        payload = yaml.safe_load(Path(item["materialized_config"]).read_text(encoding="utf-8"))
        AsterConfig(**payload["model"])
        assert len(item["config_sha256"]) == 64
    saved = json.loads((tmp_path / "campaign-manifest.json").read_text(encoding="utf-8"))
    assert saved == manifest


def test_campaign_rejects_duplicate_ids(tmp_path):
    payload = yaml.safe_load(CAMPAIGN.read_text(encoding="utf-8"))
    payload["candidates"].append(dict(payload["candidates"][0]))
    path = tmp_path / "duplicate.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        load_architecture_campaign(path, repo_root=ROOT)


def test_campaign_requires_all_fair_comparison_axes(tmp_path):
    payload = yaml.safe_load(CAMPAIGN.read_text(encoding="utf-8"))
    payload["comparison_axes"].remove("equal_cost")
    path = tmp_path / "missing-axis.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="equal_cost"):
        load_architecture_campaign(path, repo_root=ROOT)
