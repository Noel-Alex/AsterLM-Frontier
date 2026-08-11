from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from asterlm import AsterConfig, AsterLM
from scripts.run_optimizer_quality_campaign import (
    _arm_train_payload,
    _validate_resume_manifest,
    load_optimizer_campaign,
)


def test_optimizer_campaign_has_independent_family_searches():
    campaign = load_optimizer_campaign(
        "configs/experiments/optimizer_quality_campaign.yaml"
    )
    families = {arm["family"] for arm in campaign["arms"].values()}
    assert {
        "adamw-wsd",
        "adamw-cosine",
        "muon-cosine-full",
        "muon-cosine-perhead",
        "apollo-wsd",
    } <= families
    assert sum(arm["family"] == "muon-cosine-perhead" for arm in campaign["arms"].values()) >= 3
    config = AsterConfig.from_yaml(campaign["model"])
    with torch.device("meta"):
        model = AsterLM(config)
    assert model.effective_parameter_count() == pytest.approx(269_677_164, rel=5e-5)
    assert model.active_parameter_count() == pytest.approx(188_272_620, rel=5e-5)


def test_arm_payload_uses_fractional_warmup_and_metrics_only(tmp_path):
    base = yaml.safe_load(
        Path("configs/train/campaign_quality_2k_adamw.yaml").read_text(encoding="utf-8")
    )
    payload = _arm_train_payload(
        base,
        run_dir=tmp_path / "run",
        seed=7,
        max_tokens=1_638_400,
        tokenizer=tmp_path / "tokenizer.json",
        common_overrides={"micro_batch_size": 4, "gradient_accumulation_steps": 2},
        arm={
            "family": "muon-cosine-perhead",
            "warmup_fraction": 0.01,
            "train_overrides": {
                "optimizer": "muon_adamw",
                "muon_per_head": True,
                "schedule_type": "cosine",
            },
        },
        checkpoint_policy="none",
    )
    train = payload["train"]
    assert train["warmup_steps"] == 1
    assert train["optimizer"] == "muon_adamw"
    assert train["muon_per_head"] is True
    assert train["schedule_type"] == "cosine"
    assert train["checkpoint_policy"] == "none"


def test_optimizer_resume_contract_rejects_changed_arm_definition():
    existing = {
        "source_provenance": {"git_commit": "a" * 40},
        "execution_variants": ["muon"],
        "arms": {"muon": {"family": "muon", "train_overrides": {"muon_lr": 0.01}}},
        "seeds": [7],
        "tokens_per_candidate": 4096,
        "checkpoint_policy": "none",
    }
    with pytest.raises(ValueError, match="arms"):
        _validate_resume_manifest(
            existing,
            arm_ids=("muon",),
            arms={"muon": {"family": "muon", "train_overrides": {"muon_lr": 0.02}}},
            seeds=(7,),
            tokens=4096,
            checkpoint_policy="none",
        )
