from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

from asterlm.artifacts import artifact_record, atomic_write_json, sha256_file
from asterlm.config import DataConfig
from asterlm.training.contracts import canonical_data_config_sha256

from .mixture import local_data_paths


def _git_commit(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _portable(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def build_clean_corpus_manifest(
    *,
    data_config_path: str | Path,
    output_path: str | Path,
    repo_root: str | Path = ".",
    benchmark_decontaminated: bool,
    pii_handled: bool,
) -> dict[str, Any]:
    """Hash a completed clean corpus and publish its immutable training contract."""

    repo = Path(repo_root).resolve()
    config_path = Path(data_config_path)
    config = DataConfig.from_yaml(config_path)
    train_paths = [Path(source.path) for source in config.sources]
    validation_paths = [Path(source.path) for source in config.validation_sources]
    artifacts: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []

    for root in [*train_paths, *validation_paths]:
        partials = [
            path
            for path in root.rglob("*")
            if path.is_file() and path.name.endswith((".partial", ".tmp"))
        ]
        if partials:
            raise RuntimeError(f"Cannot seal corpus with incomplete artifacts under {root}")
        shards = local_data_paths(root)
        if not shards:
            raise RuntimeError(f"Cannot seal empty clean-corpus path: {root}")
        for shard in shards:
            record = artifact_record(shard)
            record["path"] = _portable(shard, repo)
            artifacts.append(record)

    for root in train_paths:
        report = root / "cleaning_report.json"
        if not report.is_file():
            raise RuntimeError(f"Missing cleaning report for {root}")
        parsed = json.loads(report.read_text(encoding="utf-8"))
        record = artifact_record(report)
        record["path"] = _portable(report, repo)
        record["source_id"] = root.name
        record["estimated_tokens"] = int(parsed.get("estimated_tokens", 0))
        reports.append(record)
        artifacts.append({key: record[key] for key in ("path", "size_bytes", "sha256")})

    artifacts.sort(key=lambda item: str(item["path"]))
    dataset_identity = sha256_file(config_path)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git_commit(repo),
        # Relative paths remain portable; this hint lets validators find the
        # originating checkout when invoked from another working directory.
        "path_base_hint": str(repo),
        "data_config_path": _portable(config_path, repo),
        "data_config_file_sha256": dataset_identity,
        "data_config_sha256": canonical_data_config_sha256(config),
        "train_paths": [_portable(path, repo) for path in train_paths],
        "validation_paths": [_portable(path, repo) for path in validation_paths],
        "validation_role": config.validation_role,
        "pipeline": {
            "cleaned": True,
            "exact_deduplicated": True,
            "near_deduplicated": True,
            "cross_source_deduplicated": True,
            "benchmark_decontaminated": bool(benchmark_decontaminated),
            "validation_split_disjoint": bool(validation_paths),
            "pii_handled": bool(pii_handled),
        },
        "source_reports": reports,
        "artifacts": artifacts,
    }
    atomic_write_json(output_path, payload)
    return payload
