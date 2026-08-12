from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_pretraining_campaign", ROOT / "scripts/run_pretraining_campaign.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_campaign_is_token_honest_and_stage_command_is_portable() -> None:
    campaign = MODULE.load_campaign(ROOT / "configs/pretraining/frontier_100b_k3.yaml")
    assert campaign["goal_tokens"] == 100_000_000_000
    assert [stage["context"] for stage in campaign["stages"]] == [8192, 16384, 32768]
    command = MODULE.stage_command(
        campaign["stages"][1],
        data=campaign["data"]["clean_config"],
        hub_repo="owner/private",
        resume=None,
        init_checkpoint="runs/stage1/checkpoint-final",
    )
    assert command[1:4] == ["scripts/studio_train.py", "--mode", "pretrain"]
    assert command[-2:] == ["--init-checkpoint", "runs/stage1/checkpoint-final"]


def test_blocked_scale_campaign_cannot_launch() -> None:
    campaign = MODULE.load_campaign(ROOT / "configs/pretraining/frontier_100b_k3.yaml")
    with pytest.raises(RuntimeError, match="not launchable"):
        MODULE.require_launchable_campaign(campaign)


def test_campaign_requires_adjacent_stage_continuation() -> None:
    campaign = MODULE.load_campaign(ROOT / "configs/pretraining/frontier_100b_k3.yaml")
    campaign["stages"][2]["init_from"] = "runs/unrelated"
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "campaign.yaml"
        path.write_text(yaml.safe_dump(campaign, sort_keys=False), encoding="utf-8")
        with pytest.raises(ValueError, match="immediately preceding"):
            MODULE.load_campaign(path)
