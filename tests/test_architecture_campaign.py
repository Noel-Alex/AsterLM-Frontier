from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from asterlm.config import AsterConfig
from asterlm.experiments import load_architecture_campaign, materialize_architecture_campaign
from scripts.run_architecture_quality_campaign import _execution_matrix, _train_payload

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "configs/experiments/architecture_campaign.yaml"


def test_project_architecture_campaign_is_valid_and_materializable(tmp_path):
    campaign = load_architecture_campaign(CAMPAIGN, repo_root=ROOT)
    assert campaign.context_lengths == (4096, 8192, 16384, 32768, 65536, 131072)
    assert campaign.candidates[0].candidate_id == "tier0-dense-mla-220m"
    assert campaign.execution_variants["cutlass-grouped"].environment == {
        "ASTER_MOE_IMPL": "cutlass"
    }
    assert campaign.execution_variants["torch-grouped"].environment == {
        "ASTER_MOE_IMPL": "torch_grouped"
    }
    manifest = materialize_architecture_campaign(campaign, tmp_path)
    assert len(manifest["candidates"]) == len(campaign.candidates)
    for item in manifest["candidates"].values():
        payload = yaml.safe_load(Path(item["materialized_config"]).read_text(encoding="utf-8"))
        AsterConfig(**payload["model"])
        assert len(item["config_sha256"]) == 64
        assert set(item["effective_variants"]) == set(item["execution_variants"])
        for variant in item["effective_variants"].values():
            variant_payload = yaml.safe_load(
                Path(variant["materialized_config"]).read_text(encoding="utf-8")
            )
            AsterConfig(**variant_payload["model"])
            assert len(variant["config_sha256"]) == 64
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


def test_campaign_rejects_unknown_execution_variant(tmp_path):
    payload = yaml.safe_load(CAMPAIGN.read_text(encoding="utf-8"))
    payload["candidates"][0]["execution_variants"].append("imaginary-backend")
    path = tmp_path / "unknown-variant.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown execution variants"):
        load_architecture_campaign(path, repo_root=ROOT)


def test_execution_matrix_is_a_real_candidate_backend_cross_product(tmp_path):
    campaign = load_architecture_campaign(CAMPAIGN, repo_root=ROOT)
    manifest = materialize_architecture_campaign(campaign, tmp_path)
    matrix = _execution_matrix(
        manifest,
        ("tier0-dense-mla-220m", "tier1-dense-kda3-mla-220m"),
        ("torch-compile-bf16", "fla-kda-compile-bf16"),
    )
    assert matrix == [
        ("tier0-dense-mla-220m", "torch-compile-bf16"),
        ("tier1-dense-kda3-mla-220m", "fla-kda-compile-bf16"),
    ]


def test_execution_matrix_rejects_variant_not_used_by_selected_candidate(tmp_path):
    campaign = load_architecture_campaign(CAMPAIGN, repo_root=ROOT)
    manifest = materialize_architecture_campaign(campaign, tmp_path)
    with pytest.raises(ValueError, match="do not apply"):
        _execution_matrix(
            manifest,
            ("tier0-dense-mla-220m",),
            ("cutlass-grouped",),
        )


def test_smoke_train_payload_avoids_duplicate_full_state_milestone(tmp_path):
    base = yaml.safe_load(
        (ROOT / "configs/train/campaign_quality_2k_adamw.yaml").read_text(encoding="utf-8")
    )
    payload = _train_payload(
        base,
        run_dir=tmp_path / "run",
        seed=7,
        max_tokens=32_768,
        tokenizer=ROOT / "artifacts/tokenizer_quality_stackfree.json",
        train_overrides={"compile": False},
        no_compile=False,
        smoke=True,
        resume=None,
    )
    train = payload["train"]
    assert train["eval_batches"] == 1
    assert train["milestone_tokens"] == []
    assert train["milestone_eval"] is False
    assert train["save_interval"] > 2
    assert train["keep_last_checkpoints"] == 1
