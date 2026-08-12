from __future__ import annotations

from pathlib import Path

import pytest

from asterlm import AsterConfig
from asterlm.generation.hub_checkpoint import (
    checkpoint_catalog,
    checkpoint_model_compatible,
    select_checkpoint_name,
    select_newer_checkpoint,
)


def test_hub_checkpoint_catalog_and_selectors() -> None:
    files = [
        "runs/stage1/checkpoints/checkpoint-00000100-tok-500/checkpoint_manifest.json",
        "runs/stage1/checkpoints/checkpoint-00000200-final/checkpoint_manifest.json",
        "runs/stage1/metrics.jsonl",
        "README.md",
    ]
    catalog = checkpoint_catalog(files)
    assert list(catalog) == ["stage1"]
    checkpoints = catalog["stage1"]
    assert select_checkpoint_name(checkpoints, "latest", latest=checkpoints[0]) == checkpoints[0]
    assert select_checkpoint_name(checkpoints, "final") == "checkpoint-00000200-final"
    assert select_checkpoint_name(checkpoints, "tokens:500") == "checkpoint-00000100-tok-500"


def test_hub_checkpoint_selector_rejects_unknown_token_milestone() -> None:
    with pytest.raises(ValueError, match="No checkpoint exists"):
        select_checkpoint_name(["checkpoint-00000001-final"], "tokens:123")


def test_checkpoint_catalog_uses_numeric_step_order_not_lexical_order() -> None:
    catalog = checkpoint_catalog(
        [
            "runs/r/checkpoints/checkpoint-9/checkpoint_manifest.json",
            "runs/r/checkpoints/checkpoint-10/checkpoint_manifest.json",
        ]
    )
    assert catalog["r"] == ["checkpoint-9", "checkpoint-10"]
    assert select_checkpoint_name(catalog["r"], "latest") == "checkpoint-10"


def test_auto_resume_prefers_greater_tokens_then_step() -> None:
    local = (Path("local"), {"tokens_seen": 1000, "step": 20})
    assert select_newer_checkpoint(local, ("remote", {"tokens_seen": 1001, "step": 1})) == "remote"
    assert select_newer_checkpoint(local, ("remote", {"tokens_seen": 1000, "step": 21})) == "remote"
    assert select_newer_checkpoint(local, ("remote", {"tokens_seen": 1000, "step": 19})) == "local"


def test_checkpoint_model_compatibility_allows_execution_only_changes() -> None:
    requested = AsterConfig(
        vocab_size=64,
        d_model=32,
        n_layers=2,
        n_heads=2,
        head_dim=16,
        ffn_hidden=64,
        max_seq_len=8192,
        kda_backend="auto",
        checkpoint_segment_size=4,
    )
    saved = AsterConfig(
        vocab_size=64,
        d_model=32,
        n_layers=2,
        n_heads=2,
        head_dim=16,
        ffn_hidden=64,
        max_seq_len=2048,
        kda_backend="torch",
        checkpoint_segment_size=1,
    )
    compatible, mismatches = checkpoint_model_compatible(requested, saved)
    assert compatible
    assert mismatches == []
    saved.d_model = 48
    compatible, mismatches = checkpoint_model_compatible(requested, saved)
    assert not compatible
    assert mismatches == ["d_model"]
