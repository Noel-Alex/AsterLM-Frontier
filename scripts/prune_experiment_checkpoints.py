#!/usr/bin/env python3
"""Audit and remove disposable experiment checkpoints without losing lineage."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from asterlm.artifacts import atomic_write_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contained(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def inventory(root: Path, *, repo_root: Path) -> list[dict[str, Any]]:
    root = root.resolve()
    runs_root = (repo_root / "runs").resolve()
    if root == runs_root or not _contained(root, runs_root):
        raise ValueError("Checkpoint pruning root must be a specific directory below runs/")

    entries: list[dict[str, Any]] = []
    for checkpoint in sorted(root.rglob("checkpoint-*")):
        resolved = checkpoint.resolve()
        if not checkpoint.is_dir() or checkpoint.is_symlink():
            continue
        if not _contained(resolved, root) or not resolved.name.startswith("checkpoint-"):
            raise RuntimeError(f"Unsafe checkpoint target: {checkpoint}")
        files = [path for path in resolved.rglob("*") if path.is_file()]
        manifest_path = resolved / "checkpoint_manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.is_file()
            else None
        )
        entries.append(
            {
                "relative_path": resolved.relative_to(repo_root).as_posix(),
                "size_bytes": sum(path.stat().st_size for path in files),
                "file_count": len(files),
                "checkpoint_manifest_sha256": (
                    _sha256(manifest_path) if manifest_path.is_file() else None
                ),
                "checkpoint_manifest": manifest,
            }
        )
    return entries


def stale_latest_pointers(
    root: Path, *, removing: set[Path] | None = None
) -> list[dict[str, str]]:
    removing = removing or set()
    pointers: list[dict[str, str]] = []
    for pointer in sorted(root.resolve().rglob("latest.txt")):
        target_name = pointer.read_text(encoding="utf-8").strip()
        target = (pointer.parent / target_name).resolve()
        if target.exists() and target not in removing:
            continue
        pointers.append(
            {
                "path": pointer.as_posix(),
                "missing_target": target.as_posix(),
            }
        )
    return pointers


def prune(
    root: Path,
    *,
    repo_root: Path,
    audit_path: Path,
    execute: bool,
) -> dict[str, Any]:
    entries = inventory(root, repo_root=repo_root)
    removing = {(repo_root / entry["relative_path"]).resolve() for entry in entries}
    stale_pointers = stale_latest_pointers(root, removing=removing)
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "executed" if execute else "dry_run",
        "root": root.resolve().relative_to(repo_root).as_posix(),
        "checkpoint_count": len(entries),
        "total_bytes": sum(int(entry["size_bytes"]) for entry in entries),
        "checkpoints": entries,
        "stale_latest_pointers": stale_pointers,
    }
    audit_path = audit_path.resolve()
    if not _contained(audit_path, repo_root.resolve()):
        raise ValueError("Audit path must remain inside the repository")
    atomic_write_json(audit_path, report)
    if execute:
        for entry in entries:
            target = (repo_root / entry["relative_path"]).resolve()
            if not _contained(target, root.resolve()) or not target.name.startswith("checkpoint-"):
                raise RuntimeError(f"Refusing unsafe checkpoint deletion: {target}")
            shutil.rmtree(target)
        for pointer in stale_pointers:
            target = Path(pointer["path"]).resolve()
            if not _contained(target, root.resolve()) or target.name != "latest.txt":
                raise RuntimeError(f"Refusing unsafe pointer deletion: {target}")
            target.unlink()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    root = args.root if args.root.is_absolute() else repo_root / args.root
    audit = args.audit if args.audit.is_absolute() else repo_root / args.audit
    report = prune(root, repo_root=repo_root, audit_path=audit, execute=args.execute)
    print(json.dumps({key: value for key, value in report.items() if key != "checkpoints"}, indent=2))


if __name__ == "__main__":
    main()
