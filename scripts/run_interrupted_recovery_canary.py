#!/usr/bin/env python3
"""Run a real SIGTERM -> full-state checkpoint -> resume laptop canary."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from asterlm.artifacts import atomic_write_json
from asterlm.source_provenance import assert_expected_checkout_source
from asterlm.training.checkpoint import resolve_checkpoint, verify_checkpoint


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "runs/promotion-canary/k3-interrupted-recovery"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def atomic_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def metric_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def data_state_schema_version(state: dict[str, Any]) -> int:
    """Read the canonical envelope while retaining legacy Studio compatibility."""

    return int(state.get("schema_version", state.get("version", -1)))


def wait_for_step(metrics: Path, process: subprocess.Popen[Any], step: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Trainer exited before interruption point: {process.returncode}")
        observed = max((int(row.get("step") or 0) for row in metric_rows(metrics)), default=0)
        if observed >= step:
            return
        time.sleep(0.5)
    raise TimeoutError(f"Trainer did not reach step {step} before timeout")


def checkpoint_audit(path: Path) -> dict[str, Any]:
    manifest = verify_checkpoint(path)
    trainer_state = torch.load(path / "trainer_state.pt", map_location="cpu", weights_only=False)
    data_state = torch.load(path / "data_state.pt", map_location="cpu", weights_only=False)
    return {
        "name": path.name,
        "manifest_sha256": sha256(path / "checkpoint_manifest.json"),
        "reason": manifest["reason"],
        "step": int(manifest["step"]),
        "tokens_seen": int(manifest["tokens_seen"]),
        "resume_state": manifest["resume_state"],
        "artifact_count": len(manifest["artifacts"]),
        "artifact_bytes": sum(int(item["size_bytes"]) for item in manifest["artifacts"]),
        "artifact_hashes": {str(item["path"]): str(item["sha256"]) for item in manifest["artifacts"]},
        "trainer_state": {
            "step": int(trainer_state["step"]),
            "tokens_seen": int(trainer_state["tokens_seen"]),
            "optimizer_state_present": bool(trainer_state.get("optimizer")),
            "rng_keys": sorted(trainer_state.get("rng", {})),
        },
        "data_state": {
            "schema_version": data_state_schema_version(data_state),
            "step": int(data_state["step"]),
            "tokens_seen": int(data_state["tokens_seen"]),
            "train_cursor_present": bool(data_state.get("train")),
        },
    }


def remove_checkpoint_payloads(run_dir: Path) -> list[str]:
    expected_root = (ROOT / "runs/promotion-canary").resolve()
    resolved = run_dir.resolve()
    try:
        resolved.relative_to(expected_root)
    except ValueError as exc:
        raise RuntimeError(f"Refusing cleanup outside {expected_root}: {resolved}") from exc
    removed = []
    for checkpoint in run_dir.glob("checkpoint-*"):
        if checkpoint.is_dir():
            shutil.rmtree(checkpoint)
            removed.append(checkpoint.name)
    (run_dir / "latest.txt").unlink(missing_ok=True)
    return sorted(removed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--interrupt-after-step", type=int, default=1)
    parser.add_argument("--final-step", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--keep-checkpoints", action="store_true")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    args = parser.parse_args()
    if args.interrupt_after_step <= 0 or args.final_step <= args.interrupt_after_step:
        raise ValueError("final step must be greater than the positive interruption step")

    source = assert_expected_checkout_source(ROOT)
    if source.get("dirty"):
        raise RuntimeError("Recovery evidence requires a clean source checkout")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Refusing to mix recovery evidence into non-empty {output}")
    output.mkdir(parents=True, exist_ok=True)
    run_dir = output / "run"
    configs = output / "configs"
    train_config = configs / "train.yaml"
    data_config = configs / "data.yaml"
    log_first = output / "interrupted.log"
    log_resume = output / "resumed.log"

    data_payload = yaml.safe_load(
        (ROOT / "runs/architecture-campaign/quality-data-100m-stackfree/data-proxy.yaml").read_text(
            encoding="utf-8"
        )
    )
    atomic_yaml(data_config, data_payload)
    atomic_yaml(
        train_config,
        {
            "train": {
                "output_dir": str(run_dir),
                "seed": 1337,
                "run_class": "exploratory",
                "deterministic_named_initialization": True,
                "device": "cuda",
                "dtype": "bfloat16",
                "matmul_precision": "high",
                "compile": False,
                "execution_backend": "aster_local",
                "execution_autotune": False,
                "moe_implementation": "cutlass",
                "precision_backend": "amp",
                "sequence_length": 2048,
                "micro_batch_size": 2,
                "gradient_accumulation_steps": 4,
                "max_steps": args.final_step,
                "optimizer": "muon_adamw",
                "muon_lr": 0.005,
                "muon_per_head": True,
                "muon_megabatch": True,
                "muon_megabatch_max_gib": 0.5,
                "adam_lr": 0.0003,
                "warmup_steps": 1,
                "schedule_type": "constant",
                "weight_decay": 0.1,
                "max_grad_norm": 1.0,
                "qk_clip_interval": 100,
                "log_interval": 1,
                "eval_interval": 100000,
                "eval_batches": 1,
                "save_interval": 100000,
                "keep_last_checkpoints": 2,
                "checkpoint_policy": "full",
                "checkpoint_pyramid_levels": 0,
                "checkpoint_local_budget_gib": 8.0,
                "num_workers": 0,
                "pin_memory": True,
                "prefetch_factor": None,
                "tokenizer_path": "artifacts/tokenizer_quality_stackfree.json",
                "tensorboard": False,
                "jsonl_metrics": True,
                "system_metrics_interval": 1.0,
                "diagnostic_interval": 100,
                "save_diagnostic_bundle": True,
                "wandb_project": args.wandb_project,
                "wandb_entity": args.wandb_entity,
                "wandb_run_name": (
                    f"{output.name}-signal-resume" if args.wandb_project else None
                ),
                "hub_repo_id": None,
                "hub_upload_every_save": False,
            }
        },
    )

    command = [
        sys.executable,
        "scripts/studio_train.py",
        "--mode",
        "pretrain",
        "--model",
        "configs/model/aster_k3_latentmoe_270m_a188m.yaml",
        "--train",
        str(train_config),
        "--data",
        str(data_config),
    ]
    environment = dict(os.environ)
    environment["ASTER_MOE_IMPL"] = "cutlass"
    environment["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    started = time.time()
    with log_first.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        wait_for_step(run_dir / "metrics.jsonl", process, args.interrupt_after_step, args.timeout)
        os.killpg(process.pid, signal.SIGTERM)
        interrupted_code = process.wait(timeout=args.timeout)
    if interrupted_code not in {130, -signal.SIGTERM}:
        raise RuntimeError(f"Graceful interruption returned {interrupted_code}, expected 130")
    stop_checkpoint = resolve_checkpoint(run_dir)
    stop_audit = checkpoint_audit(stop_checkpoint)
    if stop_audit["reason"] != "studio-stop":
        raise RuntimeError("SIGTERM did not publish a Studio stop checkpoint")
    if stop_audit["step"] < args.interrupt_after_step:
        raise RuntimeError("Stop checkpoint predates the requested boundary")

    resume_command = [*command, "--resume", str(run_dir)]
    with log_resume.open("w", encoding="utf-8") as log:
        resumed = subprocess.run(
            resume_command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=args.timeout,
            check=False,
        )
    if resumed.returncode != 0:
        raise RuntimeError(f"Resumed trainer returned {resumed.returncode}")
    final_checkpoint = resolve_checkpoint(run_dir)
    final_audit = checkpoint_audit(final_checkpoint)
    if final_audit["reason"] != "complete" or final_audit["step"] != args.final_step:
        raise RuntimeError("Resumed trainer did not reach the declared final step")
    if final_audit["tokens_seen"] <= stop_audit["tokens_seen"]:
        raise RuntimeError("Resumed trainer did not advance beyond the stop checkpoint")

    rows = metric_rows(run_dir / "metrics.jsonl")
    experiment = load_json(run_dir / "experiment.json")
    wandb_identity = experiment.get("metrics") or {}
    if args.wandb_project:
        if wandb_identity.get("wandb_project") != args.wandb_project:
            raise RuntimeError("Experiment registry lost the W&B project identity")
        if not wandb_identity.get("wandb_run_id") or not wandb_identity.get("wandb_url"):
            raise RuntimeError("W&B did not return a durable run identity and URL")
    result = {
        "schema_version": 1,
        "status": "passed",
        "created_at_unix": started,
        "finished_at_unix": time.time(),
        "source_provenance": source,
        "protocol": {
            "signal": "SIGTERM",
            "interrupt_after_step": args.interrupt_after_step,
            "final_step": args.final_step,
            "sequence_length": 2048,
            "micro_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "optimizer": "muon_adamw",
            "moe_implementation": "cutlass",
        },
        "interrupted_returncode": interrupted_code,
        "stop_checkpoint": stop_audit,
        "final_checkpoint": final_audit,
        "metrics": {
            "rows": len(rows),
            "max_step": max((int(row.get("step") or 0) for row in rows), default=0),
            "max_tokens_seen": max((int(row.get("tokens_seen") or 0) for row in rows), default=0),
            "metrics_sha256": sha256(run_dir / "metrics.jsonl"),
        },
        "logs": {
            "interrupted_sha256": sha256(log_first),
            "resumed_sha256": sha256(log_resume),
            "resume_restored_data_cursor": "restored packed buffer + source/RNG state" in log_resume.read_text(
                encoding="utf-8", errors="replace"
            ),
        },
        "wandb": {
            "entity": wandb_identity.get("wandb_entity"),
            "project": wandb_identity.get("wandb_project"),
            "run_id": wandb_identity.get("wandb_run_id"),
            "url": wandb_identity.get("wandb_url"),
            "enabled": bool(args.wandb_project),
        },
        "cleanup": {"checkpoints_removed": []},
    }
    if not result["logs"]["resume_restored_data_cursor"]:
        raise RuntimeError("Resume log does not confirm restoration of the exact local-data cursor")
    if not args.keep_checkpoints:
        result["cleanup"]["checkpoints_removed"] = remove_checkpoint_payloads(run_dir)
    atomic_write_json(output / "result.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
