#!/usr/bin/env python3
"""Create durable Hugging Face and W&B continuity evidence without training."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from asterlm.artifacts import atomic_write_json, atomic_write_text, sha256_file

ROOT = Path(__file__).resolve().parents[1]
GATES = ("huggingface_round_trip_hash", "wandb_history_resume")


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_clean() -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError("Tracking round-trip evidence requires a clean checkout")


def history_has_resume(rows: list[dict[str, Any]]) -> bool:
    phases = {
        int(row["aster_roundtrip_phase"]): int(row["_step"])
        for row in rows
        if row.get("aster_roundtrip_phase") is not None and row.get("_step") is not None
    }
    return phases.get(1) == 0 and phases.get(2) == 1


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _update_ledger(path: Path, proofs: dict[str, dict[str, str]]) -> None:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    seen: set[str] = set()
    for gate in payload.get("gates", []):
        gate_id = str(gate.get("id") or "")
        if gate_id in proofs:
            gate["status"] = "passed"
            gate["evidence"] = [proofs[gate_id]]
            gate.pop("note", None)
            seen.add(gate_id)
    if seen != set(proofs):
        raise RuntimeError(f"Promotion ledger is missing gates: {sorted(set(proofs) - seen)}")
    atomic_write_text(path, yaml.safe_dump(payload, sort_keys=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hub-repo", default="philoweeb/AsterLM-Frontier-100B")
    parser.add_argument("--wandb-project", default="asterlm-frontier")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--gates", type=Path, default=ROOT / "configs/experiments/promotion_gates.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/promotion-evidence")
    args = parser.parse_args()
    _require_clean()
    commit = _git_commit()
    nonce = uuid.uuid4().hex
    created_at = datetime.now(UTC).isoformat()

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    api.create_repo(args.hub_repo, repo_type="model", private=False, exist_ok=True)
    info = api.model_info(args.hub_repo)
    if info.private:
        api.update_repo_settings(
            repo_id=args.hub_repo,
            repo_type="model",
            private=False,
        )
        info = api.model_info(args.hub_repo)
    if info.private:
        raise RuntimeError("Checkpoint repository is not public")
    payload = {
        "schema_version": 1,
        "kind": "asterlm-hub-roundtrip-canary",
        "git_commit": commit,
        "nonce": nonce,
        "created_at_utc": created_at,
    }
    with tempfile.TemporaryDirectory(prefix="aster-tracking-roundtrip-") as folder:
        temporary = Path(folder)
        source = temporary / "canary.json"
        atomic_write_json(source, payload)
        path_in_repo = f"readiness/hub-roundtrip/{commit[:12]}-{nonce}.json"
        upload = api.upload_file(
            path_or_fileobj=source,
            path_in_repo=path_in_repo,
            repo_id=args.hub_repo,
            repo_type="model",
            commit_message="Add AsterLM readiness hash canary",
        )
        downloaded = Path(
            hf_hub_download(
                args.hub_repo,
                filename=path_in_repo,
                repo_type="model",
                revision=str(upload.oid),
                local_dir=temporary / "download",
                force_download=True,
            )
        )
        source_hash = sha256_file(source)
        downloaded_hash = sha256_file(downloaded)
        if source_hash != downloaded_hash or downloaded.read_bytes() != source.read_bytes():
            raise RuntimeError("Hugging Face round-trip changed canary bytes")

    import wandb

    run_id = f"aster-ready-{commit[:10]}-{nonce[:8]}"
    first = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        id=run_id,
        resume="allow",
        name="AsterLM tracking continuity canary",
        job_type="readiness",
        config={"git_commit": commit, "hub_repo": args.hub_repo, "canary": True},
    )
    assert first is not None
    entity = str(first.entity)
    first.log({"aster_roundtrip_phase": 1, "aster_roundtrip_value": 17}, step=0)
    first.finish()
    second = wandb.init(
        project=args.wandb_project,
        entity=entity,
        id=run_id,
        resume="must",
        name="AsterLM tracking continuity canary",
        job_type="readiness",
    )
    assert second is not None
    second.log({"aster_roundtrip_phase": 2, "aster_roundtrip_value": 23}, step=1)
    second.finish()

    run_path = f"{entity}/{args.wandb_project}/{run_id}"
    rows: list[dict[str, Any]] = []
    for _ in range(15):
        remote = wandb.Api(timeout=30).run(run_path)
        rows = list(
            remote.scan_history(
                keys=["_step", "aster_roundtrip_phase", "aster_roundtrip_value"],
                min_step=0,
                max_step=1,
            )
        )
        if history_has_resume(rows):
            break
        time.sleep(2)
    if not history_has_resume(rows):
        raise RuntimeError(f"W&B resumed history is incomplete: {rows}")

    output = args.output / commit[:12] / "tracking-roundtrip"
    output.mkdir(parents=True, exist_ok=True)
    result = output / "tracking-roundtrip-result.json"
    atomic_write_json(
        result,
        {
            "schema_version": 1,
            "status": "complete",
            "created_at_utc": created_at,
            "git_commit": commit,
            "huggingface": {
                "repo_id": args.hub_repo,
                "private": False,
                "revision": str(upload.oid),
                "path": path_in_repo,
                "source_sha256": source_hash,
                "downloaded_sha256": downloaded_hash,
                "exact_bytes": True,
            },
            "wandb": {
                "run_path": run_path,
                "run_id": run_id,
                "resumed_with_must": True,
                "history": rows,
            },
        },
    )
    artifact = {"path": _relative(result), "sha256": sha256_file(result)}
    proofs: dict[str, dict[str, str]] = {}
    for gate_id in GATES:
        proof = output / f"{gate_id}-proof.json"
        atomic_write_json(
            proof,
            {
                "schema_version": 1,
                "gate_id": gate_id,
                "status": "passed",
                "created_at_utc": datetime.now(UTC).isoformat(),
                "evaluator": {
                    "name": "asterlm-tracking-roundtrip",
                    "version": "1",
                    "git_commit": commit,
                },
                "experiment_ids": [run_id, f"hf:{args.hub_repo}@{upload.oid}"],
                "artifacts": [artifact],
            },
        )
        proofs[gate_id] = {"path": _relative(proof), "sha256": sha256_file(proof)}
    _update_ledger(args.gates, proofs)
    print(result.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
