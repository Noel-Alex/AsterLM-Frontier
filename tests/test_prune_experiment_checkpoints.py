from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.prune_experiment_checkpoints import inventory, prune


def _checkpoint(path: Path, *, tokens: int) -> Path:
    path.mkdir(parents=True)
    (path / "model.safetensors").write_bytes(b"weights")
    (path / "checkpoint_manifest.json").write_text(
        json.dumps({"schema_version": 2, "tokens_seen": tokens}), encoding="utf-8"
    )
    return path


def test_pruner_records_manifest_before_removing_payload(tmp_path: Path) -> None:
    root = tmp_path / "runs" / "campaign"
    checkpoint = _checkpoint(root / "candidate" / "checkpoint-00000001", tokens=42)
    latest = checkpoint.parent / "latest.txt"
    latest.write_text(checkpoint.name, encoding="utf-8")
    audit = root / "checkpoint-prune-audit.json"

    report = prune(root, repo_root=tmp_path, audit_path=audit, execute=True)

    assert report["mode"] == "executed"
    assert report["checkpoint_count"] == 1
    assert report["checkpoints"][0]["checkpoint_manifest"]["tokens_seen"] == 42
    assert len(report["checkpoints"][0]["checkpoint_manifest_sha256"]) == 64
    assert not checkpoint.exists()
    assert not latest.exists()
    assert len(report["stale_latest_pointers"]) == 1
    assert json.loads(audit.read_text(encoding="utf-8"))["total_bytes"] > 0


def test_pruner_refuses_workspace_wide_runs_root(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    with pytest.raises(ValueError, match="specific directory"):
        inventory(runs, repo_root=tmp_path)


def test_pruner_removes_pointer_already_made_stale(tmp_path: Path) -> None:
    root = tmp_path / "runs" / "campaign"
    root.mkdir(parents=True)
    latest = root / "latest.txt"
    latest.write_text("checkpoint-00000099", encoding="utf-8")

    report = prune(
        root,
        repo_root=tmp_path,
        audit_path=root / "pointer-audit.json",
        execute=True,
    )

    assert report["checkpoint_count"] == 0
    assert len(report["stale_latest_pointers"]) == 1
    assert not latest.exists()
