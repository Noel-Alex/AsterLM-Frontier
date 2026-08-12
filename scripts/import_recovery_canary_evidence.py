#!/usr/bin/env python3
"""Archive a passed real-signal laptop recovery canary as promotion evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = ROOT / "runs/promotion-canary/k3-interrupted-recovery-c540972/result.json"
DEFAULT_GATES = ROOT / "configs/experiments/promotion_gates.yaml"
DEFAULT_OUTPUT = ROOT / "docs/promotion-evidence"
GATE_ID = "interrupted_laptop_recovery"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def repo_relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def require_clean_checkout() -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError("Recovery evidence import must start from a clean checkout")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip().lower()


def validate_result(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("schema_version") != 1 or result.get("status") != "passed":
        raise ValueError("Recovery canary is not complete and passed")
    source = result.get("source_provenance") or {}
    if source.get("dirty") is not False or not source.get("git_commit"):
        raise ValueError("Recovery canary source is not clean and pinned")
    protocol = result.get("protocol") or {}
    if protocol.get("signal") != "SIGTERM" or protocol.get("optimizer") != "muon_adamw":
        raise ValueError("Recovery canary did not exercise the final optimizer signal path")
    if protocol.get("moe_implementation") != "cutlass":
        raise ValueError("Recovery canary used the wrong MoE backend")
    if result.get("interrupted_returncode") != 130:
        raise ValueError("Graceful interruption did not return 130")

    stop = result.get("stop_checkpoint") or {}
    final = result.get("final_checkpoint") or {}
    for label, checkpoint in (("stop", stop), ("final", final)):
        resume = checkpoint.get("resume_state") or {}
        required = {
            "model",
            "optimizer",
            "scheduler",
            "rng_python",
            "rng_numpy",
            "rng_torch_cpu",
            "rng_torch_cuda",
            "global_step",
            "tokens_seen",
            "data_pipeline",
        }
        if not all(resume.get(key) is True for key in required):
            raise ValueError(f"{label} checkpoint lacks complete resume state")
        trainer = checkpoint.get("trainer_state") or {}
        data = checkpoint.get("data_state") or {}
        if not trainer.get("optimizer_state_present") or data.get("schema_version") != 1:
            raise ValueError(f"{label} checkpoint state audit is incomplete")
        if trainer.get("step") != checkpoint.get("step") or data.get("step") != checkpoint.get("step"):
            raise ValueError(f"{label} checkpoint step identity differs across artifacts")
        if trainer.get("tokens_seen") != checkpoint.get("tokens_seen") or data.get("tokens_seen") != checkpoint.get("tokens_seen"):
            raise ValueError(f"{label} checkpoint token identity differs across artifacts")
    if stop.get("reason") != "studio-stop" or final.get("reason") != "complete":
        raise ValueError("Recovery checkpoint reasons are incorrect")
    if int(final["step"]) <= int(stop["step"]) or int(final["tokens_seen"]) <= int(stop["tokens_seen"]):
        raise ValueError("Resumed run did not advance")
    logs = result.get("logs") or {}
    if logs.get("resume_restored_data_cursor") is not True:
        raise ValueError("Exact data cursor restoration was not observed")
    removed = set((result.get("cleanup") or {}).get("checkpoints_removed") or [])
    if removed != {str(stop["name"]), str(final["name"])}:
        raise ValueError("Test checkpoint payloads were not completely cleaned up")
    return {
        "source_execution_commit": source["git_commit"],
        "signal": protocol["signal"],
        "interrupted_returncode": result["interrupted_returncode"],
        "stop_step": stop["step"],
        "stop_tokens_seen": stop["tokens_seen"],
        "final_step": final["step"],
        "final_tokens_seen": final["tokens_seen"],
        "stop_manifest_sha256": stop["manifest_sha256"],
        "final_manifest_sha256": final["manifest_sha256"],
        "stop_artifact_hashes": stop["artifact_hashes"],
        "final_artifact_hashes": final["artifact_hashes"],
        "resume_restored_data_cursor": True,
        "checkpoint_payloads_removed_after_audit": sorted(removed),
    }


def update_ledger(path: Path, proof: dict[str, str]) -> None:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for gate in payload.get("gates", []):
        if gate.get("id") == GATE_ID:
            gate["status"] = "passed"
            gate["evidence"] = [proof]
            atomic_text(path, yaml.safe_dump(payload, sort_keys=False))
            return
    raise RuntimeError(f"Promotion ledger missing {GATE_ID}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--gates", type=Path, default=DEFAULT_GATES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    commit = require_clean_checkout()
    source = load_json(args.result)
    assertions = validate_result(source)
    output = args.output / commit[:12] / "interrupted-recovery"
    output.mkdir(parents=True, exist_ok=True)
    result_copy = output / "result.json"
    shutil.copyfile(args.result, result_copy)
    summary = output / "recovery-summary.json"
    atomic_text(
        summary,
        json.dumps(
            {
                "schema_version": 1,
                "status": "passed",
                "imported_at_utc": datetime.now(timezone.utc).isoformat(),
                "importer_git_commit": commit,
                "source_result": {"path": repo_relative(result_copy), "sha256": sha256(result_copy)},
                "assertions": assertions,
            },
            indent=2,
        )
        + "\n",
    )
    artifacts = [
        {"path": repo_relative(result_copy), "sha256": sha256(result_copy)},
        {"path": repo_relative(summary), "sha256": sha256(summary)},
    ]
    proof_path = output / f"{GATE_ID}-proof.json"
    atomic_text(
        proof_path,
        json.dumps(
            {
                "schema_version": 1,
                "gate_id": GATE_ID,
                "status": "passed",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "evaluator": {
                    "name": "asterlm-recovery-canary-importer",
                    "version": "1",
                    "git_commit": commit,
                },
                "experiment_ids": ["k3-interrupted-recovery-c540972"],
                "artifacts": artifacts,
            },
            indent=2,
        )
        + "\n",
    )
    update_ledger(
        args.gates,
        {"path": repo_relative(proof_path), "sha256": sha256(proof_path)},
    )
    from asterlm.experiments import evaluate_promotion_gates

    print(json.dumps({"assertions": assertions, "promotion": evaluate_promotion_gates(args.gates).manifest()}, indent=2))


if __name__ == "__main__":
    main()
