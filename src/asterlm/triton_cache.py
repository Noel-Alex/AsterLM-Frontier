from __future__ import annotations

import json
import os
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()
_SANITIZED: set[Path] = set()


def triton_cache_root(environment: dict[str, str] | None = None) -> Path:
    source = os.environ if environment is None else environment
    configured = source.get("TRITON_CACHE_DIR")
    return Path(configured).expanduser() if configured else Path.home() / ".triton" / "cache"


def quarantine_invalid_triton_json(
    root: Path | None = None,
    *,
    minimum_age_seconds: float = 300.0,
    now: float | None = None,
    force_rescan: bool = False,
) -> list[dict[str, Any]]:
    """Quarantine stale malformed Triton metadata without racing active writers."""

    cache_root = (root or triton_cache_root()).expanduser().resolve()
    if not cache_root.is_dir():
        return []
    now = time.time() if now is None else float(now)
    with _LOCK:
        if cache_root in _SANITIZED and not force_rescan:
            return []
        repairs: list[dict[str, Any]] = []
        quarantine_root = (
            cache_root.parent / "asterlm-quarantine" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        )
        for path in cache_root.rglob("*.json"):
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            age = now - stat.st_mtime
            if age < minimum_age_seconds:
                continue
            try:
                with path.open("r", encoding="utf-8") as handle:
                    json.load(handle)
                continue
            except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
                relative = path.relative_to(cache_root)
                target = quarantine_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.replace(path, target)
                except FileNotFoundError:
                    continue
                repairs.append(
                    {
                        "source": str(path),
                        "quarantine": str(target),
                        "size_bytes": stat.st_size,
                        "age_seconds": age,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        _SANITIZED.add(cache_root)
        return repairs
