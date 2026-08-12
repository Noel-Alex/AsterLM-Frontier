from __future__ import annotations

import importlib.util
from pathlib import Path


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
