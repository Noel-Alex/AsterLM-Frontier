from pathlib import Path

import pytest

from asterlm.config import TrainConfig
from scripts.studio_train import apply_remote_durable_policy


def test_remote_durable_policy_is_five_minute_async_full_state() -> None:
    config = TrainConfig(output_dir="runs/frontier", checkpoint_interval_minutes=30.0)
    apply_remote_durable_policy(
        config,
        hub_repo="owner/private-checkpoints",
        remote_run_root=Path("/run/aster/runs"),
    )
    assert config.output_dir == "/run/aster/runs/frontier"
    assert config.checkpoint_interval_minutes == 5.0
    assert config.hub_repo_id == "owner/private-checkpoints"
    assert config.hub_upload_every_save
    assert config.hub_upload_milestones
    assert config.hub_upload_final
    assert config.hub_upload_on_stop
    assert config.hub_auto_resume_latest
    assert config.hub_include_optimizer
    assert config.hub_fail_on_error
    assert config.hub_async_upload
    assert config.hub_max_pending_uploads == 2


def test_remote_durable_policy_requires_hub_repo() -> None:
    with pytest.raises(ValueError, match="requires --hub-repo"):
        apply_remote_durable_policy(
            TrainConfig(), hub_repo=None, remote_run_root=Path("/run/aster/runs")
        )
