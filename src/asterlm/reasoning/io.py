from __future__ import annotations

import gzip
import io
import json
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

_CONTROL_JSON_NAMES = {
    "state.json",
    "manifest.json",
    "summary.json",
    "report.json",
    "audit.json",
    "metrics.json",
    "download_manifest.json",
}


def _is_control_file(path: Path) -> bool:
    lower = path.name.lower()
    if lower in _CONTROL_JSON_NAMES:
        return True
    return any(token in lower for token in ("manifest", "report", "audit", "summary", "state", "stats", "metrics"))


def iter_json_records(path: str | Path) -> Iterator[dict[str, Any]]:
    root = Path(path)
    files = sorted(root.rglob("*")) if root.is_dir() else [root]
    for item in files:
        if (
            not item.is_file()
            or item.name.endswith((".partial", ".tmp"))
            or _is_control_file(item)
        ):
            continue
        lower = item.name.lower()
        if lower.endswith(".jsonl.zst"):
            import zstandard as zstd

            raw = item.open("rb")
            stream = zstd.ZstdDecompressor().stream_reader(raw)
            handle = io.TextIOWrapper(stream, encoding="utf-8")
        elif lower.endswith(".jsonl.gz"):
            handle = gzip.open(item, "rt", encoding="utf-8")
        elif lower.endswith(".jsonl"):
            handle = item.open("r", encoding="utf-8")
        elif lower.endswith(".json"):
            payload = json.loads(item.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                for record in payload:
                    if isinstance(record, dict):
                        yield record
            elif isinstance(payload, dict):
                yield payload
            continue
        else:
            continue
        with handle:
            for line in handle:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    if isinstance(record, dict):
                        yield record


def atomic_write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return count


def append_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    return count
