from __future__ import annotations

import json

from studio.research_archive import ResearchArchive


def _write_trial(root, *, name: str, throughput: float, sequence: int = 2048) -> None:
    campaign = root / "runs" / f"campaign-{name}"
    campaign.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "ok",
        "resolved_model": {"gradient_checkpointing": True, "checkpoint_segment_size": 2},
        "resolved_train": {
            "dtype": "bfloat16",
            "precision_backend": "native_bf16",
            "optimizer": "adamw",
        },
        "system": {"gpu": {"name": "Test GPU"}},
        "architecture": {
            "effective_parameters": 200,
            "active_parameters_estimate": 100,
        },
        "summary": {
            "median_tokens_per_second": throughput,
            "measured_tokens": 8192,
            "measured_wall_time_seconds": 1.0,
            "final_memory": {"peak_allocated_gib": 4.5},
            "gpu": {"median_utilization_gpu": 91, "median_power_draw": 80},
        },
    }
    result_path = campaign / f"{name}.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    matrix_path = campaign / "matrix.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8")) if matrix_path.exists() else {
        "created_utc": "2026-08-11T00:00:00+00:00",
        "git_commit": "abc123",
        "models": {},
        "protocol": {
            "sequence": sequence,
            "batch": 2,
            "accum": 4,
            "steps": 20,
            "warmup": 5,
            "optimizer_override": "adamw",
        },
        "trials": [],
    }
    matrix["models"][name] = {"path": f"configs/{name}.yaml", "sha256": f"sha-{name}"}
    matrix["trials"].append(
        {
            "name": name,
            "variant": name,
            "status": "ok",
            "result": f"runs/campaign-{name}/{name}.json",
        }
    )
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")


def test_archive_indexes_trials_without_duplicate_revisions(tmp_path) -> None:
    _write_trial(tmp_path, name="dense", throughput=1200)
    archive = ResearchArchive(tmp_path, tmp_path / "archive.sqlite3")

    first = archive.reindex()
    second = archive.reindex()
    trials = archive.trials()

    assert first["trials"] == 1
    assert second["trials"] == 1
    assert second["trial_revisions"] == 1
    assert trials["total"] == 1
    assert trials["rows"][0]["tokens_per_second"] == 1200
    assert trials["rows"][0]["global_batch_size"] == 8
    assert archive.summary()["retention"].startswith("unbounded")


def test_archive_paginates_and_reports_comparison_mismatches(tmp_path) -> None:
    _write_trial(tmp_path, name="dense", throughput=1200, sequence=2048)
    _write_trial(tmp_path, name="moe", throughput=1800, sequence=4096)
    archive = ResearchArchive(tmp_path, tmp_path / "archive.sqlite3")
    archive.reindex()

    first_page = archive.trials(limit=1)
    second_page = archive.trials(limit=1, offset=1)
    comparison = archive.compare([first_page["rows"][0]["id"], second_page["rows"][0]["id"]])

    assert first_page["total"] == 2
    assert len(first_page["rows"]) == len(second_page["rows"]) == 1
    assert not comparison["strictly_comparable"]
    sequence = next(item for item in comparison["dimensions"] if item["field"] == "sequence_length")
    assert not sequence["match"]


def test_archive_ingests_append_only_findings_ledger(tmp_path) -> None:
    findings = tmp_path / "docs" / "research" / "findings.jsonl"
    findings.parent.mkdir(parents=True)
    findings.write_text(
        json.dumps(
            {
                "id": "utilization-repaired",
                "created_utc": "2026-08-11T00:00:00+00:00",
                "title": "Utilization repaired",
                "summary": "A larger microbatch materially raised utilization.",
                "status": "confirmed",
                "tags": ["moe", "systems"],
                "evidence": ["runs/campaign/matrix.json"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    archive = ResearchArchive(tmp_path, tmp_path / "archive.sqlite3")

    archive.reindex()
    payload = archive.findings()

    assert payload["total"] == 1
    assert payload["rows"][0]["id"] == "utilization-repaired"
    assert payload["rows"][0]["evidence"] == ["runs/campaign/matrix.json"]
