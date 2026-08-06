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


def _stack_tokens(path: str) -> int:
    raw = yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))["stack_edu"]
    return sum(int(source["target_tokens"]) for source in raw["languages"])


def test_progressive_overtrain_tiers_have_exact_token_budgets() -> None:
    assert _corpus_tokens("configs/corpus/corpus_overtrain_50b.yaml") == 43_500_000_000
    assert _stack_tokens("configs/corpus/stack_edu_6p5b.yaml") == 6_500_000_000
    assert _corpus_tokens("configs/corpus/corpus_overtrain_100b.yaml") == 87_000_000_000
    assert _stack_tokens("configs/corpus/stack_edu_13b.yaml") == 13_000_000_000


def test_100b_train_configs_sum_to_campaign_budget() -> None:
    configs = [
        TrainConfig.from_yaml(ROOT / "configs/train/frontier_100b_stage1_8k.yaml"),
        TrainConfig.from_yaml(ROOT / "configs/train/frontier_100b_stage2_16k.yaml"),
        TrainConfig.from_yaml(ROOT / "configs/train/frontier_100b_stage3_32k.yaml"),
    ]
    assert sum(config.max_tokens or 0 for config in configs) == 100_000_000_000
    assert configs[0].milestone_tokens == [18_400_000_000, 50_000_000_000, 92_000_000_000]
    assert all(config.hub_upload_milestones for config in configs)


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
