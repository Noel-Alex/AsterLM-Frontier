from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from asterlm.artifacts import atomic_write_json, sha256_file


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _git(root: Path, *args: str) -> bytes | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=15,
        )
        return result.stdout
    except Exception:
        return None


def git_provenance(root: Path) -> dict[str, Any]:
    commit_raw = _git(root, "rev-parse", "HEAD")
    status = _git(root, "status", "--porcelain=v1", "-z") or b""
    patch = _git(root, "diff", "--binary", "HEAD") or b""
    index = _git(root, "ls-files", "-s", "-z") or b""
    return {
        "git_commit": commit_raw.decode("ascii", errors="replace").strip() if commit_raw else None,
        "dirty": bool(status),
        "dirty_patch_sha256": hashlib.sha256(patch + status).hexdigest() if status else None,
        "tree_manifest_sha256": hashlib.sha256(index + status).hexdigest() if index else None,
    }


class ExperimentRegistry:
    """Atomic, machine-readable lifecycle record for one training run.

    Metrics remain append-only JSONL and checkpoints retain their own manifests. This
    record is the small authoritative index that ties those artifacts to code, data,
    environment, status, and lineage.
    """

    filename = "experiment.json"

    def __init__(self, output: str | Path, record: dict[str, Any]) -> None:
        self.output = Path(output)
        self.path = self.output / self.filename
        self.record = record

    @classmethod
    def create(
        cls,
        output: str | Path,
        *,
        repo_root: str | Path,
        model: dict[str, Any],
        train: dict[str, Any],
        data: dict[str, Any],
        environment: dict[str, Any],
        architecture: dict[str, Any],
        command: list[str] | None = None,
        parent_run_id: str | None = None,
        hypothesis: str | None = None,
        stage: str = "pretrain",
        resume_existing: bool = False,
    ) -> "ExperimentRegistry":
        output_path = Path(output)
        output_path.mkdir(parents=True, exist_ok=True)
        existing_path = output_path / cls.filename
        if existing_path.exists():
            if not resume_existing:
                raise FileExistsError(
                    f"Run directory already contains {cls.filename}: {output_path}. "
                    "Choose a new output directory or explicitly resume the existing run."
                )
            existing = json.loads(existing_path.read_text(encoding="utf-8"))
            return cls(output_path, existing)

        timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        run_id = f"{output_path.name}-{timestamp}-{uuid.uuid4().hex[:8]}"
        tokenizer = Path(str(train.get("tokenizer_path", "")))
        tokenizer_sha = sha256_file(tokenizer) if tokenizer.is_file() else None
        dataset_manifest = Path(str(data.get("manifest_path", "")))
        dataset_manifest_sha = (
            sha256_file(dataset_manifest) if dataset_manifest.is_file() else None
        )
        code = git_provenance(Path(repo_root))
        code["command"] = list(command or [])
        record: dict[str, Any] = {
            "schema_version": 1,
            "run_id": run_id,
            "parent_run_id": parent_run_id,
            "stage": stage,
            "hypothesis": hypothesis,
            "status": "planned",
            "status_reason": None,
            "started_at_utc": None,
            "ended_at_utc": None,
            "intended_tokens": train.get("max_tokens"),
            "completed_tokens": 0,
            "completion_fraction": 0.0,
            "code": code,
            "environment": {
                **environment,
                "provider": os.environ.get("ASTERLM_PROVIDER", "local"),
                "provider_profile_alias": os.environ.get("ASTERLM_PROVIDER_PROFILE"),
            },
            "data": {
                "config": data,
                "config_sha256": _canonical_sha256(data),
                "dataset_manifest_uri": data.get("manifest_path"),
                "dataset_manifest_sha256": dataset_manifest_sha,
                "tokenizer_uri": str(tokenizer) if str(tokenizer) else None,
                "tokenizer_sha256": tokenizer_sha,
                "shard_cursor": None,
            },
            "model": {
                "config": model,
                "config_sha256": _canonical_sha256(model),
                "architecture": architecture,
            },
            "train": {
                "config": train,
                "config_sha256": _canonical_sha256(train),
            },
            "metrics": {
                "metrics_jsonl_uri": "metrics.jsonl",
                "gpu_telemetry_uri": "metrics.jsonl",
                "wandb_url": None,
            },
            "resume_state": {},
            "artifacts": [],
            "notes": [],
            "updated_at_utc": _utc_now(),
        }
        registry = cls(output_path, record)
        registry.flush()
        return registry

    def flush(self) -> None:
        self.record["updated_at_utc"] = _utc_now()
        atomic_write_json(self.path, self.record)

    def mark_running(self) -> None:
        self.record["status"] = "running"
        self.record["status_reason"] = None
        self.record["started_at_utc"] = self.record.get("started_at_utc") or _utc_now()
        self.record["ended_at_utc"] = None
        self.flush()

    def update_progress(self, tokens_seen: int, **metrics: Any) -> None:
        intended = self.record.get("intended_tokens")
        self.record["completed_tokens"] = int(tokens_seen)
        self.record["completion_fraction"] = (
            min(1.0, float(tokens_seen) / float(intended)) if intended else None
        )
        self.record["metrics"].update({key: value for key, value in metrics.items() if value is not None})
        self.flush()

    def finish(self, status: str, *, tokens_seen: int, reason: str | None = None) -> None:
        self.update_progress(tokens_seen)
        self.record["status"] = status
        self.record["status_reason"] = reason
        self.record["ended_at_utc"] = _utc_now()
        self.flush()

    def add_checkpoint(self, checkpoint: str | Path, *, reason: str) -> None:
        path = Path(checkpoint)
        manifest_path = path / "checkpoint_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entry = {
            "kind": "checkpoint",
            "uri": str(path.resolve()),
            "manifest_uri": str(manifest_path.resolve()),
            "manifest_sha256": sha256_file(manifest_path),
            "size_bytes": sum(
                int(item.get("size_bytes", 0)) for item in manifest.get("artifacts", [])
            ),
            "verified": manifest.get("status") == "complete",
            "retention": "permanent" if manifest.get("permanent") else "run",
            "reason": reason,
            "step": manifest.get("step"),
            "tokens_seen": manifest.get("tokens_seen"),
        }
        artifacts = self.record.setdefault("artifacts", [])
        artifacts[:] = [item for item in artifacts if item.get("uri") != entry["uri"]]
        artifacts.append(entry)
        self.record["resume_state"] = manifest.get("resume_state", {})
        self.flush()

    def set_wandb_url(self, url: str | None) -> None:
        self.record["metrics"]["wandb_url"] = url
        self.flush()
