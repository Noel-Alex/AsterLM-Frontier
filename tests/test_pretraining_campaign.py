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
    assert [stage["context"] for stage in campaign["stages"]] == [4096, 8192, 16384, 32768]
    assert [stage["tokens"] for stage in campaign["stages"]] == [
        92_000_000_000,
        3_000_000_000,
        3_000_000_000,
        2_000_000_000,
    ]
    command = MODULE.stage_command(
        campaign["stages"][1],
        data=campaign["data"]["clean_config"],
        hub_repo="owner/private",
        resume=None,
        init_checkpoint="runs/stage1/checkpoint-final",
        promotion_gates="runs/campaign/promotion_gates.runtime.yaml",
    )
    assert command[1:4] == ["scripts/studio_train.py", "--mode", "pretrain"]
    assert command[-2:] == ["--init-checkpoint", "runs/stage1/checkpoint-final"]
    assert command[command.index("--promotion-gates") + 1].endswith(
        "promotion_gates.runtime.yaml"
    )


def test_remote_stage_command_enables_durable_ephemeral_policy() -> None:
    campaign = MODULE.load_campaign(ROOT / "configs/pretraining/frontier_100b_k3.yaml")
    command = MODULE.stage_command(
        campaign["stages"][0],
        data=campaign["data"]["clean_config"],
        hub_repo="owner/public-checkpoints",
        resume=None,
        init_checkpoint=None,
        remote_durable=True,
    )
    assert "--remote-durable" in command


def test_remote_campaign_tracks_the_effective_persistent_run_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    campaign = MODULE.load_campaign(ROOT / "configs/pretraining/frontier_100b_k3.yaml")
    monkeypatch.setenv("ASTERLM_REMOTE_RUN_ROOT", str(tmp_path / "provider-volume"))
    output = MODULE.stage_output_dir(campaign["stages"][0], remote_durable=True)
    assert output == tmp_path / "provider-volume" / "aster-frontier-100b-stage1-4k"


def test_stage_transition_prefers_a_completed_local_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    local = tmp_path / "checkpoint-final"
    monkeypatch.setattr(MODULE, "completed_checkpoint", lambda _output: local)
    monkeypatch.setattr(
        MODULE,
        "download_hub_checkpoint",
        lambda *_args, **_kwargs: pytest.fail("Hub should not be queried"),
    )
    resolved = MODULE.completed_checkpoint_or_hub(
        tmp_path,
        hub_repo="owner/public-checkpoints",
        model_path="configs/model/aster_k3_latentmoe_1p45b_a568m.yaml",
    )
    assert resolved == local


def test_frozen_scale_campaign_is_launchable() -> None:
    campaign = MODULE.load_campaign(ROOT / "configs/pretraining/frontier_100b_k3.yaml")
    MODULE.require_launchable_campaign(campaign)
    assert campaign["architecture"]["total_parameters"] == 1_448_120_880
    assert campaign["architecture"]["active_parameters_per_token"] == 568_155_376


def test_stage_environment_is_explicit_and_allowlisted() -> None:
    campaign = MODULE.load_campaign(ROOT / "configs/pretraining/frontier_100b_k3.yaml")
    environment = MODULE.stage_environment({"WANDB_ENTITY": "owner"}, campaign["stages"][2])
    assert environment["WANDB_ENTITY"] == "owner"
    assert environment["FLA_DISABLE_BACKEND_DISPATCH"] == "1"
    campaign["stages"][2]["environment"]["HF_TOKEN"] = "must-not-live-in-campaign"
    with pytest.raises(ValueError, match="unsupported environment"):
        MODULE.stage_environment({}, campaign["stages"][2])


def test_campaign_requires_adjacent_stage_continuation() -> None:
    campaign = MODULE.load_campaign(ROOT / "configs/pretraining/frontier_100b_k3.yaml")
    campaign["stages"][3]["init_from"] = "runs/unrelated"
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "campaign.yaml"
        path.write_text(yaml.safe_dump(campaign, sort_keys=False), encoding="utf-8")
        with pytest.raises(ValueError, match="immediately preceding"):
            MODULE.load_campaign(path)


def test_runtime_promotion_ledger_is_resumable_and_does_not_mutate_canonical(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical.yaml"
    canonical.write_text(
        "schema_version: 2\nrepo_root: ../..\ngates:\n"
        "- id: long_context_retrieval\n  status: not_run\n"
        "- id: correctness_and_data_quality_clear\n  status: not_run\n",
        encoding="utf-8",
    )
    state_root = tmp_path / "runs" / "campaign"
    runtime = MODULE.initialize_runtime_promotion_ledger(
        state_root,
        canonical=canonical,
    )
    assert runtime.read_text(encoding="utf-8") == canonical.read_text(encoding="utf-8")
    runtime_payload = yaml.safe_load(runtime.read_text(encoding="utf-8"))
    runtime_payload["gates"][0]["status"] = "passed"
    runtime_payload["gates"][0]["evidence"] = [{"path": "runs/proof.json", "sha256": "a" * 64}]
    runtime.write_text(yaml.safe_dump(runtime_payload, sort_keys=False), encoding="utf-8")
    canonical_payload = yaml.safe_load(canonical.read_text(encoding="utf-8"))
    canonical_payload["gates"][1]["status"] = "passed"
    canonical_payload["gates"][1]["evidence"] = [{"path": "docs/data.json", "sha256": "b" * 64}]
    canonical.write_text(yaml.safe_dump(canonical_payload, sort_keys=False), encoding="utf-8")
    refreshed = yaml.safe_load(MODULE.initialize_runtime_promotion_ledger(
        state_root,
        canonical=canonical,
    ).read_text(encoding="utf-8"))
    assert refreshed["gates"][0]["status"] == "passed"
    assert refreshed["gates"][1]["status"] == "passed"
    assert yaml.safe_load(canonical.read_text(encoding="utf-8"))["gates"][0]["status"] == "not_run"


def test_long_context_transition_commands_bind_checkpoint_and_runtime_ledger(
    tmp_path: Path,
) -> None:
    campaign = MODULE.load_campaign(ROOT / "configs/pretraining/frontier_100b_k3.yaml")
    checkpoint = tmp_path / "provider-volume" / "stage1" / "checkpoint-final"
    evaluate, promote = MODULE.long_context_gate_command(
        gate=MODULE.LONG_CONTEXT_GATES[0],
        prior_stage=campaign["stages"][0],
        checkpoint=checkpoint,
        state_root=tmp_path / "runs" / "campaign",
        promotion_gates=tmp_path / "runs" / "campaign" / "promotion_gates.runtime.yaml",
    )
    assert evaluate[evaluate.index("--checkpoint") + 1] == str(checkpoint)
    assert evaluate[evaluate.index("--lengths") + 1] == "8192,16384,32768"
    assert promote[promote.index("--expected-checkpoint-root") + 1] == str(
        checkpoint.parent
    )
    assert promote[promote.index("--gates") + 1].endswith(
        "promotion_gates.runtime.yaml"
    )
    assert MODULE.LONG_CONTEXT_GATES[-1]["gate"] == "final_long_context_retrieval"
    assert MODULE.LONG_CONTEXT_GATES[-1]["required_before_index"] == 4
    assert MODULE.LONG_CONTEXT_GATES[-1]["lengths"] == "65536,131072,262144"
