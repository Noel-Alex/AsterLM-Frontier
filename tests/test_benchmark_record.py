from __future__ import annotations

import csv
import json

import pytest

from asterlm.experiments import (
    MANDATORY_BENCHMARK_FIELDS,
    BenchmarkRecordError,
    validate_benchmark_record,
    write_benchmark_csv,
    write_benchmark_record,
)


def complete_record():
    record = {field: 1.0 for field in MANDATORY_BENCHMARK_FIELDS}
    record.update(
        {
            "architecture_id": "dense-mla-666m",
            "git_sha": "abc123",
            "config_hash": "def456",
            "backend": "ada-sm89",
            "gpu": "RTX 4080 Laptop GPU",
            "precision": "bf16",
            "domain_validation_losses": {"code": 2.1, "math": 2.3},
            "long_context_scores": {"8192": 0.95},
            "benchmark_scores": {"hellaswag": 0.42},
        }
    )
    return record


def test_complete_benchmark_record_round_trip(tmp_path):
    path = tmp_path / "benchmark.json"
    normalized = write_benchmark_record(path, complete_record())
    assert json.loads(path.read_text(encoding="utf-8")) == normalized
    assert normalized["schema_version"] == 1


def test_partial_record_is_explicitly_allowed_only_during_collection():
    record = complete_record()
    record["energy_per_token"] = None
    assert validate_benchmark_record(record, require_complete=False)["energy_per_token"] is None
    with pytest.raises(BenchmarkRecordError, match="energy_per_token"):
        validate_benchmark_record(record)


@pytest.mark.parametrize(
    ("field", "value"),
    [("mean_gpu_util", 101), ("wall_time", -1), ("validation_loss", float("nan"))],
)
def test_invalid_measurements_are_rejected(field, value):
    record = complete_record()
    record[field] = value
    with pytest.raises(BenchmarkRecordError):
        validate_benchmark_record(record)


def test_csv_export_keeps_nested_scores_machine_readable(tmp_path):
    path = tmp_path / "benchmarks.csv"
    write_benchmark_csv(path, [complete_record()])
    with path.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert json.loads(row["domain_validation_losses"])["code"] == 2.1
