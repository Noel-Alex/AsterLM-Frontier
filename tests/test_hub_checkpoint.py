from __future__ import annotations

import pytest

from asterlm.generation.hub_checkpoint import checkpoint_catalog, select_checkpoint_name


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
