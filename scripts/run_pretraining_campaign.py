#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from asterlm.artifacts import atomic_write_json
from asterlm.training.checkpoint import resolve_checkpoint, verify_checkpoint

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_STAGE_ENVIRONMENT = {
    "FLA_DISABLE_BACKEND_DISPATCH",
    "PYTORCH_ALLOC_CONF",
}


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
    if resume:
        command.extend(["--resume", resume])
    elif init_checkpoint:
        command.extend(["--init-checkpoint", init_checkpoint])
    return command


def completed_checkpoint(output_dir: str | Path) -> Path | None:
    root = ROOT / output_dir
    if not root.is_dir():
        return None
    checkpoint = resolve_checkpoint(root)
    if checkpoint == root or not checkpoint.is_dir():
        return None
    manifest = verify_checkpoint(checkpoint)
    return checkpoint if manifest.get("reason") == "complete" else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen Aster K3 pretraining stages unattended")
    parser.add_argument(
        "--campaign",
        default="configs/pretraining/frontier_100b_k3.yaml",
    )
    parser.add_argument("--hub-repo", required=True, help="Private namespace/repository")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--start-stage", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-manifest-hashes", action="store_true")
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
    state: dict[str, Any] = {
        "schema_version": 1,
        "campaign": str(campaign_path.relative_to(ROOT)),
        "name": campaign["name"],
        "goal_tokens": campaign["goal_tokens"],
        "hub_repo": args.hub_repo,
        "status": "dry_run" if args.dry_run else "preflight",
        "created_at_unix": time.time(),
        "stages": [],
    }

    commands: list[dict[str, Any]] = []
    previous_complete: Path | None = None
    for stage in campaign["stages"]:
        complete = completed_checkpoint(stage["output_dir"])
        if complete is not None:
            previous_complete = complete
        if stage not in stages:
            continue
        run_root = ROOT / stage["output_dir"]
        resume = str(run_root) if run_root.is_dir() and resolve_checkpoint(run_root) != run_root else None
        init_checkpoint = (
            None
            if resume or not stage.get("init_from")
            else str(previous_complete or (ROOT / str(stage["init_from"])))
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
        run_root = ROOT / stage["output_dir"]
        resume = str(run_root) if run_root.is_dir() and resolve_checkpoint(run_root) != run_root else None
        init_checkpoint: str | None = None
        if stage.get("init_from") and not resume:
            if previous_complete is None:
                previous_complete = completed_checkpoint(str(stage["init_from"]))
            if previous_complete is None:
                raise RuntimeError(f"{stage['id']} requires the completed prior stage checkpoint")
            init_checkpoint = str(previous_complete)
        command = stage_command(
            stage,
            data=data,
            hub_repo=args.hub_repo,
            resume=resume,
            init_checkpoint=init_checkpoint,
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
        checkpoint = completed_checkpoint(stage["output_dir"])
        if checkpoint is None:
            record["status"] = "failed_missing_complete_checkpoint"
            state["status"] = "failed"
            atomic_write_json(state_path, state)
            raise RuntimeError(f"{stage['id']} exited successfully without a complete checkpoint")
        record["status"] = "complete"
        record["checkpoint"] = str(checkpoint.relative_to(ROOT))
        previous_complete = checkpoint
        atomic_write_json(state_path, state)
    state["status"] = "complete"
    state["current_stage"] = None
    state["finished_at_unix"] = time.time()
    atomic_write_json(state_path, state)


if __name__ == "__main__":
    main()
