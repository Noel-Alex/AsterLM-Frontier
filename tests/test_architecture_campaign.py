from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

from asterlm import AsterLM
from asterlm.config import AsterConfig
from asterlm.experiments import load_architecture_campaign, materialize_architecture_campaign
from asterlm.layers.gdn2 import gdn2_is_available
from scripts.run_architecture_quality_campaign import (
    _absolute_data_config,
    _execution_matrix,
    _train_payload,
)

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "configs/experiments/architecture_campaign.yaml"


def test_project_architecture_campaign_is_valid_and_materializable(tmp_path):
    campaign = load_architecture_campaign(CAMPAIGN, repo_root=ROOT)
    assert campaign.context_lengths == (4096, 8192, 16384, 32768, 65536, 131072)
    assert campaign.candidates[0].candidate_id == "tier0-dense-mla-220m"
    assert campaign.execution_variants["cutlass-grouped"].train_overrides[
        "moe_implementation"
    ] == "cutlass"
    assert campaign.execution_variants["cutlass-grouped"].environment == {}
    assert campaign.execution_variants["cutlass-grouped"].comparison_role == "system_recipe"
    assert campaign.execution_variants["cutlass-grouped-bf16"].train_overrides[
        "precision_backend"
    ] == "amp"
    assert (
        campaign.execution_variants["cutlass-grouped-bf16"].numerical_family
        == campaign.execution_variants["torch-reference"].numerical_family
    )
    assert campaign.execution_variants["torch-grouped"].train_overrides[
        "moe_implementation"
    ] == "torch_grouped"
    assert campaign.execution_variants["liger-experts-bf16"].train_overrides[
        "moe_implementation"
    ] == "liger"
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


def test_quality_train_config_promotes_moe_environment_to_first_class_field(tmp_path):
    base = yaml.safe_load(Path("configs/train/campaign_quality_2k_adamw.yaml").read_text())
    payload = _train_payload(
        base,
        run_dir=tmp_path / "run",
        seed=1337,
        max_tokens=1_048_576,
        tokenizer=Path("artifacts/tokenizer_proxy.json"),
        train_overrides={"compile": False},
        environment={"ASTER_MOE_IMPL": "cutlass"},
        no_compile=False,
        smoke=True,
        resume=None,
    )
    assert payload["train"]["moe_implementation"] == "cutlass"


def test_quality_execution_data_paths_are_independent_of_pinned_checkout(tmp_path):
    source = tmp_path / "train.jsonl"
    source.write_text('{"text":"hello"}\n', encoding="utf-8")
    data_path = tmp_path / "data.yaml"
    data_path.write_text(
        yaml.safe_dump(
            {"data": {"sources": [{"path": "train.jsonl", "weight": 1.0}]}},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    target = tmp_path / "generated" / "absolute.yaml"
    generated = _absolute_data_config(data_path, tmp_path, target)
    payload = yaml.safe_load(generated.read_text(encoding="utf-8"))
    assert payload["data"]["sources"][0]["path"] == str(source.resolve())


def test_execution_matrix_rejects_variant_not_used_by_selected_candidate(tmp_path):
    campaign = load_architecture_campaign(CAMPAIGN, repo_root=ROOT)
    manifest = materialize_architecture_campaign(campaign, tmp_path)
    with pytest.raises(ValueError, match="do not apply"):
        _execution_matrix(
            manifest,
            ("tier0-dense-mla-220m",),
            ("cutlass-grouped",),
        )


def test_smoke_train_payload_is_metrics_only_by_default(tmp_path):
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
    assert train["checkpoint_policy"] == "none"


@pytest.mark.parametrize(
    ("path", "expected_total", "expected_active"),
    [
        ("aster_k3_latentmoe_868m_a483m.yaml", 868_300_000, 483_400_000),
        ("aster_k3_latentmoe_1p45b_a568m.yaml", 1_448_100_000, 568_200_000),
        ("aster_k3_latentmoe_1p95b_a766m.yaml", 1_954_400_000, 765_900_000),
    ],
)
def test_k3_scale_frontier_parameter_geometry(path, expected_total, expected_active):
    config = AsterConfig.from_yaml(ROOT / "configs" / "model" / path)
    # Parameter accounting is backend-independent. CPU CI intentionally omits
    # FLA, so instantiate the readable recurrence without weakening the pinned
    # production config itself.
    config = replace(config, kda_backend="torch")
    with torch.device("meta"):
        model = AsterLM(config)
    assert model.effective_parameter_count() == pytest.approx(expected_total, rel=5e-4)
    assert model.active_parameter_count() == pytest.approx(expected_active, rel=5e-4)


def test_final_challengers_isolate_mixer_and_sparse_capacity() -> None:
    incumbent_config = AsterConfig.from_yaml(
        ROOT / "configs/model/aster_k3_latentmoe_1p45b_a568m.yaml"
    )
    gdn2_config = AsterConfig.from_yaml(
        ROOT / "configs/model/aster_gdn2_latentmoe_selected_body.yaml"
    )
    dense_config = AsterConfig.from_yaml(
        ROOT / "configs/model/aster_dense_kda3_mla_final_control.yaml"
    )
    total_matched_dense_config = AsterConfig.from_yaml(
        ROOT / "configs/model/aster_dense_kda3_mla_total_matched_control.yaml"
    )
    assert gdn2_config.pattern == ["gdn2", "gdn2", "gdn2", "latent"] * 8
    assert incumbent_config.pattern == ["kda", "kda", "kda", "latent"] * 8
    assert gdn2_config.ffn_type == incumbent_config.ffn_type == "latent_moe"
    assert dense_config.pattern == incumbent_config.pattern
    assert dense_config.ffn_type == "dense"

    with torch.device("meta"):
        incumbent = AsterLM(replace(incumbent_config, kda_backend="torch"))
        dense = AsterLM(replace(dense_config, kda_backend="torch"))
        total_matched_dense = AsterLM(
            replace(total_matched_dense_config, kda_backend="torch")
        )
    # GDN2's mixer itself is larger while the MoE body remains byte-for-byte
    # geometrically identical. Equal-active-FLOP analysis controls that 12.4%
    # difference; changing experts to hide it would confound the mixer test.
    if gdn2_is_available():
        with torch.device("meta"):
            gdn2 = AsterLM(gdn2_config)
        assert gdn2.effective_parameter_count() == 1_518_591_264
        assert gdn2.active_parameter_count() == 638_625_760
    assert dense.active_parameter_count() == pytest.approx(
        incumbent.active_parameter_count(), rel=0.01
    )
    assert total_matched_dense.effective_parameter_count() == 1_445_907_696
    assert total_matched_dense.effective_parameter_count() == pytest.approx(
        incumbent.effective_parameter_count(), rel=0.002
    )
    assert total_matched_dense.active_parameter_count() == (
        total_matched_dense.effective_parameter_count()
    )


def test_total_matched_mtp_candidates_add_identical_jointly_trained_capacity(tmp_path):
    campaign = load_architecture_campaign(CAMPAIGN, repo_root=ROOT)
    manifest = materialize_architecture_campaign(campaign, tmp_path)
    ids = (
        "tier2-k3-latentmoe-1p45b-mtp1",
        "tier2-dense-kda3-mla-total-matched-mtp1",
    )
    configs = {
        candidate_id: AsterConfig.from_yaml(
            Path(manifest["candidates"][candidate_id]["materialized_config"])
        )
        for candidate_id in ids
    }
    with torch.device("meta"):
        moe = AsterLM(replace(configs[ids[0]], kda_backend="torch"))
        dense = AsterLM(replace(configs[ids[1]], kda_backend="torch"))
    assert configs[ids[0]].mtp_depth == configs[ids[1]].mtp_depth == 1
    assert configs[ids[0]].mtp_loss_weight == configs[ids[1]].mtp_loss_weight == 0.12
    assert moe.effective_parameter_count() == 1_449_105_200
    assert dense.effective_parameter_count() == 1_446_892_016
    assert moe.effective_parameter_count() - 1_448_120_880 == 984_320
    assert dense.effective_parameter_count() - 1_445_907_696 == 984_320
    assert dense.effective_parameter_count() == pytest.approx(
        moe.effective_parameter_count(), rel=0.002
    )


def test_stage1_aligned_4k_variant_preserves_logical_batch(tmp_path):
    campaign = load_architecture_campaign(CAMPAIGN, repo_root=ROOT)
    manifest = materialize_architecture_campaign(campaign, tmp_path)
    variant = manifest["execution_variants"][
        "cutlass-grouped-k3-muon8bit-bf16-4k"
    ]["train_overrides"]
    assert variant["sequence_length"] == 4096
    assert variant["micro_batch_size"] == 1
    assert variant["gradient_accumulation_steps"] == 4
    assert (
        variant["sequence_length"]
        * variant["micro_batch_size"]
        * variant["gradient_accumulation_steps"]
        == 16_384
    )
    assert variant["eval_batches"] * variant["sequence_length"] == 16_384


def test_csa_hca_proxy_is_parameter_matched_to_dense_mla(tmp_path):
    campaign = load_architecture_campaign(CAMPAIGN, repo_root=ROOT)
    manifest = materialize_architecture_campaign(campaign, tmp_path)
    configs = {}
    for candidate_id in ("tier0-dense-mla-220m", "tier4-dense-csa-hca-220m"):
        materialized = Path(manifest["candidates"][candidate_id]["materialized_config"])
        configs[candidate_id] = AsterConfig.from_yaml(materialized)

    with torch.device("meta"):
        dense = AsterLM(configs["tier0-dense-mla-220m"])
        compressed = AsterLM(configs["tier4-dense-csa-hca-220m"])
    assert compressed.active_parameter_count() == pytest.approx(
        dense.active_parameter_count(), rel=3e-3
    )


def test_mhc_proxy_pays_for_mixers_by_reducing_ffn_width(tmp_path):
    campaign = load_architecture_campaign(CAMPAIGN, repo_root=ROOT)
    manifest = materialize_architecture_campaign(campaign, tmp_path)
    models = {}
    for candidate_id in ("tier0-dense-mla-220m", "tier5-dense-mla-mhc-220m"):
        materialized = Path(manifest["candidates"][candidate_id]["materialized_config"])
        with torch.device("meta"):
            models[candidate_id] = AsterLM(AsterConfig.from_yaml(materialized))
    assert models["tier5-dense-mla-mhc-220m"].active_parameter_count() == pytest.approx(
        models["tier0-dense-mla-220m"].active_parameter_count(), rel=2e-4
    )
