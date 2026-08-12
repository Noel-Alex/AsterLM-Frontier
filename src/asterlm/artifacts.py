from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Hash an artifact without loading it into RAM."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(path: str | Path, *, relative_to: str | Path | None = None) -> dict[str, Any]:
    artifact = Path(path)
    name = artifact.name
    if relative_to is not None:
        try:
            name = artifact.relative_to(Path(relative_to)).as_posix()
        except ValueError:
            pass
    return {
        "path": name,
        "size_bytes": artifact.stat().st_size,
        "sha256": sha256_file(artifact),
    }


def fsync_file(path: str | Path) -> None:
    """Ask the OS to persist an already-written file before publishing a pointer."""
    with Path(path).open("rb") as handle:
        os.fsync(handle.fileno())


def fsync_directory(path: str | Path) -> None:
    """Persist directory metadata where the platform exposes directory handles."""
    if os.name == "nt":
        return
    descriptor = os.open(Path(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_text(path: str | Path, text: str, *, encoding: str = "utf-8") -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("w", encoding=encoding, newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: str | Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=False, default=str) + "\n")
