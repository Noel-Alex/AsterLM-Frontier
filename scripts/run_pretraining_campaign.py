#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from asterlm.artifacts import atomic_write_json
from asterlm.config import AsterConfig
from asterlm.generation.hub_checkpoint import (
    checkpoint_model_compatible,
    download_hub_checkpoint,
)
from asterlm.training.checkpoint import resolve_checkpoint, verify_checkpoint

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_STAGE_ENVIRONMENT = {
    "FLA_DISABLE_BACKEND_DISPATCH",
    "PYTORCH_ALLOC_CONF",
}
LONG_CONTEXT_GATES = (
    {
        "gate": "long_context_retrieval",
        "prior_stage_index": 0,
        "required_before_index": 1,
        "lengths": "8192,16384,32768",
    },
    {
        "gate": "stage2_long_context_retrieval",
        "prior_stage_index": 1,
        "required_before_index": 2,
        "lengths": "16384,32768,65536",
    },
    {
        "gate": "stage3_long_context_retrieval",
        "prior_stage_index": 2,
        "required_before_index": 3,
        "lengths": "32768,65536,131072",
    },
)


def load_campaign(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if payload.get("schema_version") != 1 or not payload.get("stages"):
        raise ValueError("Pretraining campaign requires schema_version=1 and stages")
    if sum(int(stage["tokens"]) for stage in payload["stages"]) != int(payload["goal_tokens"]):
        raise ValueError("Campaign stage tokens do not sum to goal_tokens")
    for index, stage in enumerate(payload["stages"]):
        if index == 0:
            if stage.get("init_from") is not None:
                raise ValueError("The first pretraining stage cannot declare init_from")
            continue
        previous = payload["stages"][index - 1]
        if str(stage.get("init_from")) != str(previous.get("output_dir")):
            raise ValueError(
                f"{stage['id']} must initialize from the immediately preceding "
                f"stage output {previous.get('output_dir')!r}"
            )
    return payload


def require_launchable_campaign(campaign: dict[str, Any]) -> None:
    if campaign.get("status") != "ready":
        raise RuntimeError(
            "Pretraining campaign is not launchable: "
            f"status={campaign.get('status')!r}. Complete scale selection and "
            "write one architecture-compatible model family into every stage first."
        )
    architecture = campaign.get("architecture") or {}
    base_model = architecture.get("base_model")
    if not base_model:
        raise RuntimeError("Launchable campaign has no selected architecture.base_model")
    for stage in campaign["stages"]:
        if not stage.get("model"):
            raise RuntimeError(f"Launchable campaign stage {stage['id']} has no model")


def stage_environment(base: dict[str, str], stage: dict[str, Any]) -> dict[str, str]:
    overrides = stage.get("environment") or {}
    if not isinstance(overrides, dict):
        raise TypeError(f"{stage['id']} environment must be a mapping")
    unsupported = set(overrides) - ALLOWED_STAGE_ENVIRONMENT
    if unsupported:
        raise ValueError(
            f"{stage['id']} has unsupported environment override(s): "
            f"{', '.join(sorted(unsupported))}"
        )
    environment = dict(base)
    environment.update({str(key): str(value) for key, value in overrides.items()})
    return environment


def child_process_options() -> dict[str, Any]:
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def forward_signal(child: subprocess.Popen[Any], signum: int) -> None:
    if child.poll() is not None:
        return
    if os.name == "nt":
        event = signal.CTRL_BREAK_EVENT if signum == signal.SIGINT else signal.CTRL_C_EVENT
        child.send_signal(event)
    else:
        os.killpg(child.pid, signum)


def stage_command(
    stage: dict[str, Any],
    *,
    data: str,
    hub_repo: str,
    resume: str | None,
    init_checkpoint: str | None,
    promotion_gates: str | Path | None = None,
    remote_durable: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/studio_train.py",
        "--mode",
        "pretrain",
        "--model",
        str(stage["model"]),
        "--train",
        str(stage["train"]),
        "--data",
        data,
        "--hub-repo",
        hub_repo,
    ]
    if promotion_gates is not None:
        command.extend(["--promotion-gates", str(promotion_gates)])
    if remote_durable:
        command.append("--remote-durable")
    if resume:
        command.extend(["--resume", resume])
    elif init_checkpoint:
        command.extend(["--init-checkpoint", init_checkpoint])
    return command


def initialize_runtime_promotion_ledger(
    state_root: Path,
    *,
    canonical: Path = ROOT / "configs/experiments/promotion_gates.yaml",
) -> Path:
    """Create a resumable, ignored gate ledger without dirtying frozen source."""

    ledger = state_root / "promotion_gates.runtime.yaml"
    if ledger.is_file():
        return ledger
    state_root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(canonical, ledger)
    return ledger


def promotion_gate_passed(ledger: Path, gate_id: str) -> bool:
    payload = yaml.safe_load(ledger.read_text(encoding="utf-8")) or {}
    matches = [gate for gate in payload.get("gates", []) if gate.get("id") == gate_id]
    if len(matches) != 1:
        raise RuntimeError(f"Promotion ledger has {len(matches)} matches for {gate_id}")
    return matches[0].get("status") == "passed"


def long_context_gate_command(
    *,
    gate: dict[str, Any],
    prior_stage: dict[str, Any],
    checkpoint: Path,
    state_root: Path,
    promotion_gates: Path,
) -> tuple[list[str], list[str]]:
    raw_output = state_root / "long-context-evaluations" / str(gate["gate"])
    evaluate = [
        sys.executable,
        "scripts/long_context_retrieval.py",
        "--checkpoint",
        str(checkpoint),
        "--model",
        str(prior_stage["model"]),
        "--moe-implementation",
        "cutlass",
        "--lengths",
        str(gate["lengths"]),
        "--depths",
        "0.1,0.5,0.9",
        "--tasks",
        "exact_key,repeated_key,two_hop",
        "--repeats",
        "3",
        "--output",
        str(raw_output),
    ]
    promote = [
        sys.executable,
        "scripts/import_long_context_evidence.py",
        "--summary",
        str(raw_output / "summary.json"),
        "--gate",
        str(gate["gate"]),
        "--gates",
        str(promotion_gates),
        "--output",
        str(state_root / "promotion-evidence"),
        "--expected-checkpoint-root",
        str(checkpoint.parent),
    ]
    return evaluate, promote


def completed_checkpoint(output_dir: str | Path) -> Path | None:
    root = Path(output_dir)
    if not root.is_absolute():
        root = ROOT / root
    if not root.is_dir():
        return None
    checkpoint = resolve_checkpoint(root)
    if checkpoint == root or not checkpoint.is_dir():
        return None
    manifest = verify_checkpoint(checkpoint)
    return checkpoint if manifest.get("reason") == "complete" else None


def stage_output_dir(stage: dict[str, Any], *, remote_durable: bool) -> Path:
    """Resolve the output directory exactly as ``studio_train.py`` will."""

    configured = Path(str(stage["output_dir"]))
    if remote_durable:
        remote_root = Path(os.environ.get("ASTERLM_REMOTE_RUN_ROOT", "/opt/aster/runs"))
        return remote_root / configured.name
    return configured if configured.is_absolute() else ROOT / configured


def completed_checkpoint_or_hub(
    output_dir: str | Path,
    *,
    hub_repo: str,
    model_path: str | Path,
) -> Path:
    """Resolve a completed prior stage locally or from its verified Hub final."""

    local = completed_checkpoint(output_dir)
    if local is not None:
        return local
    run = Path(output_dir).name
    checkpoint, _record = download_hub_checkpoint(
        hub_repo,
        run=run,
        selector="final",
    )
    manifest = verify_checkpoint(checkpoint)
    if manifest.get("reason") != "complete":
        raise RuntimeError(f"Hub final for {run} is not a completed stage checkpoint")
    requested = AsterConfig.from_yaml(ROOT / model_path)
    saved = AsterConfig.from_yaml(checkpoint / "model_config.yaml")
    compatible, mismatches = checkpoint_model_compatible(requested, saved)
    if not compatible:
        raise RuntimeError(
            f"Hub final for {run} is incompatible with the prior stage model: {mismatches}"
        )
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen Aster K3 pretraining stages unattended")
    parser.add_argument(
        "--campaign",
        default="configs/pretraining/frontier_100b_k3.yaml",
    )
    parser.add_argument("--hub-repo", required=True, help="Public namespace/repository")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--start-stage", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-manifest-hashes", action="store_true")
    parser.add_argument(
        "--remote-durable",
        action="store_true",
        help=(
            "Use the ephemeral-provider contract: five-minute full-state saves, "
            "verified public Hub upload at every save, and newest-Hub auto-resume"
        ),
    )
    args = parser.parse_args()

    campaign_path = (ROOT / args.campaign).resolve()
    campaign = load_campaign(campaign_path)
    if not args.dry_run:
        require_launchable_campaign(campaign)
    stages = list(campaign["stages"])
    if args.start_stage:
        indices = [index for index, stage in enumerate(stages) if stage["id"] == args.start_stage]
        if not indices:
            raise ValueError(f"Unknown campaign stage: {args.start_stage}")
        stages = stages[indices[0] :]
    data = str(campaign["data"]["clean_config"])
    state_root = ROOT / "runs" / f"{campaign['name']}-campaign"
    state_path = state_root / "campaign_state.json"
    runtime_promotion_gates = state_root / "promotion_gates.runtime.yaml"
    state: dict[str, Any] = {
        "schema_version": 1,
        "campaign": str(campaign_path.relative_to(ROOT)),
        "name": campaign["name"],
        "goal_tokens": campaign["goal_tokens"],
        "hub_repo": args.hub_repo,
        "promotion_gates": str(runtime_promotion_gates),
        "status": "dry_run" if args.dry_run else "preflight",
        "created_at_unix": time.time(),
        "stages": [],
    }

    commands: list[dict[str, Any]] = []
    previous_complete: Path | None = None
    for stage in campaign["stages"]:
        effective_output = stage_output_dir(stage, remote_durable=args.remote_durable)
        complete = completed_checkpoint(effective_output)
        if complete is not None:
            previous_complete = complete
        if stage not in stages:
            continue
        run_root = effective_output
        resume = str(run_root) if run_root.is_dir() and resolve_checkpoint(run_root) != run_root else None
        init_checkpoint = (
            None
            if resume or not stage.get("init_from")
            else str(
                previous_complete
                or stage_output_dir(
                    {"output_dir": stage["init_from"]},
                    remote_durable=args.remote_durable,
                )
            )
        )
        commands.append(
            {
                "stage": stage,
                "command": stage_command(
                    stage,
                    data=data,
                    hub_repo=args.hub_repo,
                    resume=resume,
                    init_checkpoint=init_checkpoint,
                    promotion_gates=runtime_promotion_gates,
                    remote_durable=args.remote_durable,
                ),
            }
        )
        if complete is not None:
            previous_complete = complete

    if args.dry_run:
        state["commands"] = [
            {
                "stage": item["stage"]["id"],
                "command": item["command"],
                "environment": item["stage"].get("environment") or {},
            }
            for item in commands
        ]
        atomic_write_json(state_path, state)
        print(json.dumps(state, indent=2))
        return

    first = commands[0]["stage"]
    runtime_promotion_gates = initialize_runtime_promotion_ledger(state_root)
    preflight = [
        sys.executable,
        "scripts/training_preflight.py",
        "--model",
        str(first["model"]),
        "--train",
        str(first["train"]),
        "--data",
        data,
        "--hub-repo",
        args.hub_repo,
        "--check-first-record",
        "--json",
        str(state_root / "preflight.json"),
    ]
    if args.verify_manifest_hashes:
        preflight.append("--verify-manifest-hashes")
    state_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(state_path, state)
    subprocess.run(preflight, cwd=ROOT, check=True)

    environment = dict(os.environ)
    if args.wandb_entity:
        environment["WANDB_ENTITY"] = args.wandb_entity
    child: subprocess.Popen[Any] | None = None

    def forward(signum: int, _frame: Any) -> None:
        if child is not None and child.poll() is None:
            forward_signal(child, signum)

    signal.signal(signal.SIGINT, forward)
    signal.signal(signal.SIGTERM, forward)
    state["status"] = "running"
    for item in commands:
        stage = item["stage"]
        stage_index = campaign["stages"].index(stage)

        # A fresh provider may begin at stage 3 or 4, so materialize every
        # prerequisite proof, not only the immediately preceding transition.
        for gate in LONG_CONTEXT_GATES:
            if int(gate["required_before_index"]) > stage_index:
                continue
            gate_id = str(gate["gate"])
            if promotion_gate_passed(runtime_promotion_gates, gate_id):
                continue
            prior_stage = campaign["stages"][int(gate["prior_stage_index"])]
            prior_checkpoint = completed_checkpoint_or_hub(
                stage_output_dir(prior_stage, remote_durable=args.remote_durable),
                hub_repo=args.hub_repo,
                model_path=prior_stage["model"],
            )
            evaluate, promote = long_context_gate_command(
                gate=gate,
                prior_stage=prior_stage,
                checkpoint=prior_checkpoint,
                state_root=state_root,
                promotion_gates=runtime_promotion_gates,
            )
            gate_record = {
                "id": gate_id,
                "status": "running",
                "checkpoint": str(prior_checkpoint),
                "evaluate_command": evaluate,
                "promote_command": promote,
                "started_at_unix": time.time(),
            }
            state.setdefault("stage_transition_evaluations", []).append(gate_record)
            state["current_transition_evaluation"] = gate_id
            atomic_write_json(state_path, state)
            evaluation_environment = stage_environment(environment, prior_stage)
            for command in (evaluate, promote):
                child = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=evaluation_environment,
                    **child_process_options(),
                )
                gate_record["pid"] = child.pid
                atomic_write_json(state_path, state)
                code = child.wait()
                if code != 0:
                    gate_record["status"] = "failed"
                    gate_record["returncode"] = code
                    gate_record["finished_at_unix"] = time.time()
                    state["status"] = "failed_transition_evaluation"
                    atomic_write_json(state_path, state)
                    raise SystemExit(code)
            if not promotion_gate_passed(runtime_promotion_gates, gate_id):
                raise RuntimeError(f"{gate_id} importer exited without passing its gate")
            gate_record["status"] = "passed"
            gate_record["returncode"] = 0
            gate_record["finished_at_unix"] = time.time()
            state["current_transition_evaluation"] = None
            atomic_write_json(state_path, state)

        run_root = stage_output_dir(stage, remote_durable=args.remote_durable)
        resume = str(run_root) if run_root.is_dir() and resolve_checkpoint(run_root) != run_root else None
        init_checkpoint: str | None = None
        if stage.get("init_from") and not resume:
            if previous_complete is None:
                stage_index = campaign["stages"].index(stage)
                prior_stage = campaign["stages"][stage_index - 1]
                previous_complete = completed_checkpoint_or_hub(
                    stage_output_dir(prior_stage, remote_durable=args.remote_durable),
                    hub_repo=args.hub_repo,
                    model_path=prior_stage["model"],
                )
            init_checkpoint = str(previous_complete)
        command = stage_command(
            stage,
            data=data,
            hub_repo=args.hub_repo,
            resume=resume,
            init_checkpoint=init_checkpoint,
            promotion_gates=runtime_promotion_gates,
            remote_durable=args.remote_durable,
        )
        record = {
            "id": stage["id"],
            "status": "running",
            "command": command,
            "started_at_unix": time.time(),
        }
        state["current_stage"] = stage["id"]
        state["stages"].append(record)
        atomic_write_json(state_path, state)
        child = subprocess.Popen(
            command,
            cwd=ROOT,
            env=stage_environment(environment, stage),
            **child_process_options(),
        )
        record["pid"] = child.pid
        atomic_write_json(state_path, state)
        code = child.wait()
        record["returncode"] = code
        record["finished_at_unix"] = time.time()
        if code != 0:
            record["status"] = "stopped" if code == 130 else "failed"
            state["status"] = record["status"]
            atomic_write_json(state_path, state)
            raise SystemExit(code)
        checkpoint = completed_checkpoint(run_root)
        if checkpoint is None:
            record["status"] = "failed_missing_complete_checkpoint"
            state["status"] = "failed"
            atomic_write_json(state_path, state)
            raise RuntimeError(f"{stage['id']} exited successfully without a complete checkpoint")
        record["status"] = "complete"
        record["checkpoint"] = str(checkpoint)
        previous_complete = checkpoint
        atomic_write_json(state_path, state)
    state["status"] = "complete"
    state["current_stage"] = None
    state["finished_at_unix"] = time.time()
    atomic_write_json(state_path, state)


if __name__ == "__main__":
    main()
