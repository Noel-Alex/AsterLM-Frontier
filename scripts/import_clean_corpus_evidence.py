#!/usr/bin/env python3
"""Validate, hash, and promote the final cleaned pretraining corpus."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from asterlm.artifacts import atomic_write_json, atomic_write_text, sha256_file
from asterlm.config import DataConfig
from asterlm.training.contracts import validate_clean_manifest

ROOT = Path(__file__).resolve().parents[1]
GATE_ID = "correctness_and_data_quality_clear"
DEFAULT_SOURCES = (
    "fineweb_edu",
    "dclm",
    "cosmopedia_v2",
    "finemath_4plus",
    "nemotron_math",
)


def _require_clean() -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError("Clean-corpus evidence import requires a clean checkout")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def validate_corpus(
    data_path: Path,
    *,
    expected_sources: set[str],
    minimum_clean_tokens: int,
) -> dict[str, Any]:
    data = DataConfig.from_yaml(data_path)
    manifest_path, manifest = validate_clean_manifest(data, verify_artifact_hashes=True)
    train_by_id = {Path(source.path).name: source for source in data.sources}
    validation_by_id = {Path(source.path).name: source for source in data.validation_sources}
    if set(train_by_id) != expected_sources or set(validation_by_id) != expected_sources:
        raise ValueError(
            "Clean source ids do not match the declared campaign: "
            f"train={sorted(train_by_id)}, validation={sorted(validation_by_id)}"
        )
    for label, sources in (("train", data.sources), ("validation", data.validation_sources)):
        total = sum(float(source.weight) for source in sources)
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"{label} mixture weights sum to {total}, not 1")
    for source_id, source in train_by_id.items():
        if float(source.fim_rate) > 0 and "code" not in source_id:
            raise ValueError(f"Non-code source {source_id!r} has FIM enabled")
    reports = {str(row["source_id"]): row for row in manifest.get("source_reports", [])}
    if set(reports) != expected_sources:
        raise ValueError(f"Manifest source reports differ: {sorted(reports)}")

    total_tokens = 0
    source_results: dict[str, Any] = {}
    for source_id in sorted(expected_sources):
        root = Path(train_by_id[source_id].path)
        cleaning_path = root / "cleaning_report.json"
        audit_path = root / "audit_report.json"
        cleaning = json.loads(cleaning_path.read_text(encoding="utf-8"))
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        arguments = cleaning.get("arguments") or {}
        counts = cleaning.get("counts") or {}
        if str(arguments.get("source_id")) != source_id:
            raise ValueError(f"Cleaning report source mismatch for {source_id}")
        if not arguments.get("benchmark"):
            raise ValueError(f"Source {source_id} was cleaned without benchmark decontamination")
        if float(arguments.get("validation_fraction") or 0.0) <= 0:
            raise ValueError(f"Source {source_id} has no deterministic validation split")
        if int(counts.get("train_kept") or 0) <= 0 or int(counts.get("validation_kept") or 0) <= 0:
            raise ValueError(f"Source {source_id} has an empty train or validation split")
        estimated_tokens = int(cleaning.get("estimated_tokens") or 0)
        if estimated_tokens <= 0 or estimated_tokens != int(reports[source_id]["estimated_tokens"]):
            raise ValueError(f"Source {source_id} has inconsistent token accounting")
        total_records = int(audit.get("total_records_seen") or 0)
        sample_size = int(audit.get("sample_size") or 0)
        if sample_size != min(10_000, total_records):
            raise ValueError(f"Source {source_id} does not have the required audit sample")
        if int(audit.get("detected_pii_items_before_redaction") or 0) != 0:
            raise ValueError(f"Source {source_id} audit still detects PII")
        decisions = audit.get("quality_decisions") or {}
        invalid = sum(int(value) for key, value in decisions.items() if key != "ok")
        if invalid:
            raise ValueError(f"Source {source_id} audit rejected {invalid} cleaned records")
        total_tokens += estimated_tokens
        source_results[source_id] = {
            "estimated_clean_tokens": estimated_tokens,
            "train_records": int(counts["train_kept"]),
            "validation_records": int(counts["validation_kept"]),
            "audit_sample_size": sample_size,
            "cleaning_report_sha256": sha256_file(cleaning_path),
            "audit_report_sha256": sha256_file(audit_path),
        }
    if total_tokens < minimum_clean_tokens:
        raise ValueError(f"Clean unique-token estimate {total_tokens:,} is below {minimum_clean_tokens:,}")
    return {
        "manifest": {
            "path": manifest_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(manifest_path),
        },
        "data_config": {"path": data_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(data_path)},
        "pipeline": manifest["pipeline"],
        "expected_sources": sorted(expected_sources),
        "estimated_clean_tokens": total_tokens,
        "maximum_effective_epochs_for_100b": 100_000_000_000 / total_tokens,
        "sources": source_results,
    }


def _update_ledger(path: Path, proof: dict[str, str]) -> None:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for gate in payload.get("gates", []):
        if gate.get("id") == GATE_ID:
            gate["status"] = "passed"
            gate["evidence"] = [proof]
            atomic_write_text(path, yaml.safe_dump(payload, sort_keys=False))
            return
    raise RuntimeError(f"Promotion ledger is missing {GATE_ID}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/clean-frontier/pretrain_data.yaml")
    parser.add_argument("--expected-source", action="append", default=[])
    parser.add_argument("--minimum-clean-tokens", type=int, default=50_000_000_000)
    parser.add_argument("--gates", type=Path, default=ROOT / "configs/experiments/promotion_gates.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/promotion-evidence")
    args = parser.parse_args()
    commit = _require_clean()
    expected = set(args.expected_source or DEFAULT_SOURCES)
    result = validate_corpus(
        args.data.resolve(),
        expected_sources=expected,
        minimum_clean_tokens=args.minimum_clean_tokens,
    )
    output = args.output / commit[:12] / "clean-corpus"
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "clean-corpus-result.json"
    atomic_write_json(
        result_path,
        {
            "schema_version": 1,
            "status": "passed",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "git_commit": commit,
            "assertions": result,
        },
    )
    artifact = {"path": result_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(result_path)}
    proof_path = output / f"{GATE_ID}-proof.json"
    atomic_write_json(
        proof_path,
        {
            "schema_version": 1,
            "gate_id": GATE_ID,
            "status": "passed",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "evaluator": {"name": "asterlm-clean-corpus-importer", "version": "1", "git_commit": commit},
            "experiment_ids": [f"clean-corpus-{commit[:12]}"],
            "artifacts": [artifact, result["manifest"], result["data_config"]],
        },
    )
    _update_ledger(
        args.gates,
        {"path": proof_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(proof_path)},
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
