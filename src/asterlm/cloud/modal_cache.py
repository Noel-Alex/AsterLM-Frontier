from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any

from asterlm.training.contracts import REQUIRED_CLEANING_FLAGS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside_data_root(path: Path, data_root: Path) -> tuple[Path, str]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(data_root.resolve())
    except ValueError as exc:
        raise ValueError(f"Modal dataset cache files must stay under {data_root}: {path}") from exc
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"Unsafe Modal dataset cache path: {path}")
    return resolved, PurePosixPath(*relative.parts).as_posix()


def _manifest_path(raw: str, *, manifest: dict[str, Any], root: Path) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    hint = Path(str(manifest.get("path_base_hint") or "")).expanduser()
    base = hint if hint.is_dir() else root
    return (base / path).resolve()


def build_modal_cache_stage_plan(
    manifest_path: str | Path,
    *,
    root: str | Path,
    profile_alias: str,
    modal_environment: str,
    volume_name: str,
    volume_version: int = 2,
    verify_local_hashes: bool = False,
) -> dict[str, Any]:
    """Build a no-network plan containing only sealed clean-corpus artifacts."""

    root = Path(root).resolve()
    data_root = root / "data"
    manifest_path, manifest_remote = _inside_data_root(Path(manifest_path), data_root)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    pipeline = payload.get("pipeline")
    missing = sorted(
        flag
        for flag in REQUIRED_CLEANING_FLAGS
        if not isinstance(pipeline, dict) or pipeline.get(flag) is not True
    )
    if payload.get("schema_version") != 1 or payload.get("status") != "complete" or missing:
        raise ValueError(
            "Only a complete decision-grade clean-corpus manifest may be staged; "
            f"missing guarantees: {missing}"
        )
    records = payload.get("artifacts")
    if not isinstance(records, list) or not records:
        raise ValueError("Clean-corpus manifest has no sealed artifacts")

    requested: list[tuple[Path, str, int, str, str]] = []
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("Malformed clean-corpus artifact record")
        local = _manifest_path(str(record.get("path") or ""), manifest=payload, root=root)
        local, remote = _inside_data_root(local, data_root)
        expected_size = int(record.get("size_bytes", -1))
        expected_sha = str(record.get("sha256") or "")
        if not local.is_file() or local.stat().st_size != expected_size or len(expected_sha) != 64:
            raise ValueError(f"Clean-corpus artifact is missing or changed: {local}")
        if verify_local_hashes and _sha256(local) != expected_sha:
            raise ValueError(f"Clean-corpus artifact hash changed: {local}")
        requested.append((local, remote, expected_size, expected_sha, "artifact"))

    config_raw = str(payload.get("data_config_path") or "")
    if not config_raw:
        raise ValueError("Clean-corpus manifest has no data_config_path")
    config = _manifest_path(config_raw, manifest=payload, root=root)
    config, config_remote = _inside_data_root(config, data_root)
    if not config.is_file():
        raise ValueError(f"Clean-corpus data config is missing: {config}")
    config_sha = _sha256(config)
    expected_config_file_sha = str(payload.get("data_config_file_sha256") or "")
    if expected_config_file_sha and config_sha != expected_config_file_sha:
        raise ValueError("Clean-corpus data config differs from its sealed manifest")
    requested.append((config, config_remote, config.stat().st_size, config_sha, "data_config"))

    manifest_sha = _sha256(manifest_path)
    requested.append(
        (manifest_path, manifest_remote, manifest_path.stat().st_size, manifest_sha, "commit_marker")
    )
    files: dict[str, dict[str, Any]] = {}
    for local, remote, size, sha, role in requested:
        current = files.get(remote)
        row = {
            "local_path": str(local),
            "volume_path": remote,
            "size_bytes": size,
            "sha256": sha,
            "role": role,
        }
        if current is not None and current["sha256"] != sha:
            raise ValueError(f"Conflicting sealed artifacts target {remote}")
        files[remote] = row
    ordered = sorted(
        files.values(), key=lambda row: (row["role"] == "commit_marker", row["volume_path"])
    )
    return {
        "schema_version": 1,
        "provider": "modal",
        "status": "ready",
        "profile_alias": profile_alias,
        "modal_environment": modal_environment,
        "volume_name": volume_name,
        "volume_version": int(volume_version),
        "manifest": {
            "local_path": str(manifest_path),
            "volume_path": manifest_remote,
            "sha256": manifest_sha,
        },
        "files": ordered,
        "file_count": len(ordered),
        "total_bytes": sum(int(row["size_bytes"]) for row in ordered),
        "raw_corpus_included": False,
        "upload_policy": (
            "Volume v2 content-addressed batch upload; the clean manifest is the final "
            "atomic commit marker and reruns reuse unchanged blocks"
        ),
        "local_hashes_verified": bool(verify_local_hashes),
    }


async def _read_remote_file(volume, path: str, *, maximum_bytes: int) -> bytes:
    result = bytearray()
    async for chunk in volume.read_file(path):
        result.extend(chunk)
        if len(result) > maximum_bytes:
            raise RuntimeError(f"Remote cache file is unexpectedly large: {path}")
    return bytes(result)


def execute_modal_cache_stage(plan: dict[str, Any]) -> dict[str, Any]:
    """Populate one profile Volume; never creates a Sandbox or requests a GPU."""

    if plan.get("provider") != "modal" or plan.get("status") != "ready":
        raise RuntimeError("Refusing a blocked or non-Modal cache-stage plan")
    if not plan.get("local_hashes_verified"):
        raise RuntimeError("Execute requires a plan with full local artifact hash verification")
    if os.environ.get("MODAL_PROFILE") != plan.get("profile_alias"):
        raise RuntimeError("MODAL_PROFILE does not match the cache-stage plan")

    import modal
    from modal.exception import Error as ModalError

    volume = modal.Volume.from_name(
        str(plan["volume_name"]),
        environment_name=str(plan["modal_environment"]),
        create_if_missing=True,
        version=int(plan["volume_version"]),
    )
    marker = plan["manifest"]
    try:
        current = asyncio.run(
            _read_remote_file(
                volume,
                str(marker["volume_path"]),
                maximum_bytes=max(int(plan["files"][-1]["size_bytes"]), 1) + 1,
            )
        )
    except (ModalError, OSError, RuntimeError):
        current = None
    if current is not None and hashlib.sha256(current).hexdigest() == marker["sha256"]:
        return {
            "status": "already_current",
            "volume_name": plan["volume_name"],
            "manifest_sha256": marker["sha256"],
            "uploaded_files": 0,
        }

    # Volume v2 batches use content hashes/block hashes. Re-running an interrupted
    # stage therefore reuses blocks already present instead of retransmitting them.
    with volume.batch_upload(force=False) as batch:
        for row in plan["files"]:
            batch.put_file(row["local_path"], "/" + row["volume_path"])
    return {
        "status": "staged",
        "volume_name": plan["volume_name"],
        "manifest_sha256": marker["sha256"],
        "uploaded_files": plan["file_count"],
        "logical_bytes": plan["total_bytes"],
    }
