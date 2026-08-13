from __future__ import annotations

from pathlib import Path

import torch
import yaml

from asterlm.config import AsterConfig, TrainConfig
from asterlm.model import AsterLM

ROOT = Path(__file__).resolve().parents[1]


def _corpus_tokens(path: str) -> int:
    raw = yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))["corpus"]
    return sum(int(source["target_tokens"]) for source in raw["sources"])


def test_active_overtrain_tranches_exclude_retired_stack_edu() -> None:
    assert _corpus_tokens("configs/corpus/corpus_overtrain_50b.yaml") == 43_500_000_000
    assert _corpus_tokens("configs/corpus/corpus_overtrain_100b.yaml") == 87_000_000_000
    source = (ROOT / "scripts/download_data.py").read_text(encoding="utf-8")
    assert "frontier-stack-edu" not in source
    assert "overtrain100-stack-edu" not in source


def test_nemotron_candidate_pool_is_separate_and_revision_pinned() -> None:
    path = ROOT / "configs/corpus/corpus_nemotron_candidates_16b.yaml"
    corpus = yaml.safe_load(path.read_text(encoding="utf-8"))["corpus"]
    assert sum(int(source["target_tokens"]) for source in corpus["sources"]) == 16_000_000_000
    assert corpus["output_dir"] == "data/corpus-nemotron-candidates"
    assert all(len(source["revision"]) == 40 for source in corpus["sources"])


def test_100b_train_configs_sum_to_campaign_budget() -> None:
    configs = [
        TrainConfig.from_yaml(ROOT / "configs/train/frontier_100b_stage1_4k.yaml"),
        TrainConfig.from_yaml(ROOT / "configs/train/frontier_100b_stage2_8k.yaml"),
        TrainConfig.from_yaml(ROOT / "configs/train/frontier_100b_stage3_16k.yaml"),
        TrainConfig.from_yaml(ROOT / "configs/train/frontier_100b_stage4_32k.yaml"),
    ]
    assert sum(config.max_tokens or 0 for config in configs) == 100_000_000_000
    assert len(configs[0].milestone_tokens) == 13
    assert configs[0].milestone_tokens[-3:] == [65_000_000_000, 80_000_000_000, 92_000_000_000]
    assert sum(len(config.milestone_tokens) for config in configs) == 28
    assert all(config.checkpoint_policy == "full" for config in configs)
    assert [config.promotion_phase for config in configs] == [
        "stage1",
        "stage2",
        "stage3",
        "stage4",
    ]
    assert all(config.checkpoint_pyramid_levels == 8 for config in configs)
    assert [config.checkpoint_interval_minutes for config in configs] == [30.0, 30.0, 30.0, 5.0]
    assert all(config.save_interval == 250 for config in configs)
    assert all(
        config.sequence_length
        * config.micro_batch_size
        * config.gradient_accumulation_steps
        * config.save_interval
        == 32_768_000
        for config in configs
    )
    assert all(config.hub_upload_milestones for config in configs)
    assert all(not config.activation_offload for config in configs)
    assert all(config.optimizer != "torchao_cpu_offload_adamw" for config in configs)
    assert all(
        config.loqt_merge_interval == 0 or not config.loqt_merge_on_cpu
        for config in configs
    )

    campaign = yaml.safe_load(
        (ROOT / "configs/pretraining/frontier_100b_k3.yaml").read_text(encoding="utf-8")
    )
    assert campaign["status"] == "ready"
    assert campaign["architecture"]["base_model"].endswith("1p45b_a568m.yaml")
    assert campaign["architecture"]["total_parameters"] == 1_448_120_880
    assert campaign["architecture"]["active_parameters_per_token"] == 568_155_376
    selection = yaml.safe_load(
        (ROOT / "configs/experiments/pretraining_selection.yaml").read_text(encoding="utf-8")
    )["selection"]
    assert selection["status"] == "frozen"
    assert selection["model"].endswith("1p45b_a568m.yaml")
    assert len(selection["scale_finalists"]) == 3
    checkpointing = campaign["checkpointing"]
    assert checkpointing["huggingface_hard_cap_tb_decimal"] == 7.5
    assert checkpointing["huggingface_operational_guard_tb_decimal"] == 7.0
    assert checkpointing["permanent_checkpoint_count"] == 28
    assert checkpointing["projected_remote_permanent_checkpoint_gib"] == 168.0
    assert "refuse" in checkpointing["remote_quota_policy"]
    assert checkpointing["laptop_recovery_interval_minutes"] == 30
    assert checkpointing["cloud_recovery_interval_minutes"] == 5


def test_long_context_inference_configs_are_parameter_compatible_and_bounded() -> None:
    base = AsterConfig.from_yaml(
        ROOT / "configs/model/aster_k3_latentmoe_1p45b_a568m.yaml"
    )
    context_256k = AsterConfig.from_yaml(
        ROOT / "configs/model/aster_k3_latentmoe_1p45b_a568m_longctx.yaml"
    )
    context_1m = AsterConfig.from_yaml(
        ROOT / "configs/model/aster_k3_latentmoe_1p45b_a568m_1m_inference.yaml"
    )
    parameter_fields = (
        "vocab_size",
        "d_model",
        "n_layers",
        "n_heads",
        "head_dim",
        "moe_num_experts",
        "moe_top_k",
        "moe_shared_experts",
        "moe_expert_hidden",
        "latent_moe_dim",
        "kda_num_heads",
        "kda_head_dim",
        "latent_rank",
        "rope_dim",
    )
    assert all(
        getattr(base, field) == getattr(context_256k, field) == getattr(context_1m, field)
        for field in parameter_fields
    )
    assert context_256k.max_seq_len == 262_144
    assert context_1m.max_seq_len == 1_048_576
    assert context_256k.attention_window == context_1m.attention_window == 8192
    assert context_1m.rope_scaling_factor == 128.0


def test_moe_pathway_telemetry_is_finite() -> None:
    config = AsterConfig(
        vocab_size=128,
        d_model=32,
        n_layers=4,
        n_heads=4,
        head_dim=8,
        ffn_hidden=64,
        ffn_type="moe",
        moe_first_dense_layers=0,
        moe_num_experts=4,
        moe_top_k=2,
        moe_shared_experts=1,
        moe_expert_hidden=48,
        max_seq_len=16,
        kda_ratio=0,
        latent_rank=8,
        rope_dim=8,
        attention_window=16,
        sink_tokens=0,
        mtp_depth=0,
        gradient_checkpointing=False,
    )
    model = AsterLM(config)
    model(torch.randint(0, config.vocab_size, (1, 8)), return_logits=False)
    stats = model.moe_pathway_stats(sample_tokens=8)
    assert stats["moe_path_tokens_sampled"] == 8
    assert stats["moe_path_layers"] == 4
    for key, value in stats.items():
        assert torch.isfinite(torch.tensor(value)), key
