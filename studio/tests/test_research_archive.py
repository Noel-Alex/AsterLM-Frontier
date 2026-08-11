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


def test_archive_indexes_quality_runs_and_aggregate_metrics(tmp_path) -> None:
    campaign = tmp_path / "runs" / "quality"
    run = campaign / "seed-7" / "k3" / "muon"
    adam_run = campaign / "seed-7" / "k3" / "adamw"
    run.mkdir(parents=True)
    adam_run.mkdir(parents=True)
    experiment = {
        "status": "ok",
        "started_at_utc": "2026-08-11T00:00:00Z",
        "code": {"git_commit": "abc123"},
        "environment": {"gpu": {"name": "Test GPU"}},
        "model": {
            "config_sha256": "model-sha",
            "config": {"gradient_checkpointing": True, "checkpoint_segment_size": 2},
            "architecture": {
                "effective_parameters": 270,
                "active_parameters_estimate": 188,
            },
        },
        "train": {
            "config_sha256": "train-sha",
            "config": {
                "sequence_length": 2048,
                "micro_batch_size": 4,
                "gradient_accumulation_steps": 2,
                "warmup_steps": 10,
                "optimizer": "muon_adamw",
                "dtype": "bfloat16",
                "precision_backend": "amp",
            },
        },
    }
    (run / "experiment.json").write_text(json.dumps(experiment), encoding="utf-8")
    adam_experiment = json.loads(json.dumps(experiment))
    adam_experiment["train"]["config"]["optimizer"] = "adamw"
    adam_experiment["train"]["config"]["warmup_steps"] = 5
    (adam_run / "experiment.json").write_text(
        json.dumps(adam_experiment), encoding="utf-8"
    )
    quality_campaign = {
        "campaign_type": "optimizer_quality",
        "comparison_contract": {
            "treatment_fields": ["optimizer", "warmup_steps"]
        },
        "source_provenance": {"git_commit": "abc123"},
        "runs": {
            "7:k3:muon": {
                "model_config": "configs/k3.yaml",
                "model_config_sha256": "model-sha",
            },
            "7:k3:adamw": {
                "model_config": "configs/k3.yaml",
                "model_config_sha256": "model-sha",
            },
        },
    }
    (campaign / "quality-campaign.json").write_text(
        json.dumps(quality_campaign), encoding="utf-8"
    )
    analysis = {
        "runs": [
            {
                "seed": 7,
                "candidate_id": "k3",
                "execution_variant": "muon",
                "run_dir": "seed-7/k3/muon",
                "status": "ok",
                "tokens_seen": 4096,
                "eval_main_loss": 3.5,
                "median_training_tokens_per_second": 1200,
                "mean_gpu_util_percent": 91,
                "peak_vram_gib": 4.5,
                "wall_clock_total_seconds": 10,
                "learning_curve": [
                    {"step": 1, "tokens_seen": 2048, "eval_main_loss": 4.0},
                    {"step": 2, "tokens_seen": 4096, "eval_main_loss": 3.5},
                ],
            },
            {
                "seed": 7,
                "candidate_id": "k3",
                "execution_variant": "adamw",
                "run_dir": "seed-7/k3/adamw",
                "status": "ok",
                "tokens_seen": 4096,
                "eval_main_loss": 3.6,
                "median_training_tokens_per_second": 1250,
                "mean_gpu_util_percent": 92,
                "peak_vram_gib": 4.4,
                "wall_clock_total_seconds": 9.5,
                "learning_curve": [
                    {"step": 1, "tokens_seen": 2048, "eval_main_loss": 4.1},
                    {"step": 2, "tokens_seen": 4096, "eval_main_loss": 3.6},
                ],
            },
        ],
        "candidates": {
            "k3:muon": {
                "candidate_id": "k3",
                "execution_variant": "muon",
                "complete_seed_count": 1,
                "expected_seed_count": 1,
                "final_eval_loss_mean": 3.5,
                "final_eval_loss_stdev": None,
                "token_curve_auc_mean": 3.75,
                "wall_curve_auc_mean": 3.75,
                "equal_wall_loss_mean": 3.5,
                "equal_active_flops_loss_mean": 3.5,
                "time_to_common_loss_seconds_mean": 10,
                "tokens_to_common_loss_mean": 4096,
                "active_flops_to_common_loss_mean": 1000000,
                "median_training_tokens_per_second": 1200,
                "mean_gpu_util_percent": 91,
                "peak_vram_gib": 4.5,
            },
            "k3:adamw": {
                "candidate_id": "k3",
                "execution_variant": "adamw",
                "complete_seed_count": 1,
                "expected_seed_count": 1,
                "final_eval_loss_mean": 3.6,
                "token_curve_auc_mean": 3.85,
                "wall_curve_auc_mean": 3.85,
                "equal_wall_loss_mean": 3.6,
                "equal_active_flops_loss_mean": 3.6,
                "time_to_common_loss_seconds_mean": 9.5,
                "tokens_to_common_loss_mean": 4096,
                "active_flops_to_common_loss_mean": 1000000,
                "median_training_tokens_per_second": 1250,
                "mean_gpu_util_percent": 92,
                "peak_vram_gib": 4.4,
            },
        },
    }
    (campaign / "quality-analysis.json").write_text(json.dumps(analysis), encoding="utf-8")

    archive = ResearchArchive(tmp_path, tmp_path / "archive.sqlite3")
    indexed = archive.reindex()
    rows = archive.trials(query="muon")["rows"]

    assert indexed["trials"] == 4
    assert len(rows) == 2
    aggregate = next(row for row in rows if row["seed"] is None)
    individual = next(row for row in rows if row["seed"] == 7)
    assert aggregate["quality_trial"] == 1
    assert aggregate["eval_loss"] == 3.5
    assert aggregate["token_curve_auc"] == 3.75
    assert aggregate["time_to_common_loss"] == 10
    assert individual["result_path"] == "runs/quality/seed-7/k3/muon/experiment.json"
    assert individual["total_parameters"] == 270
    assert individual["active_parameters"] == 188

    aggregates = [row for row in archive.trials(limit=10)["rows"] if row["seed"] is None]
    comparison = archive.compare([row["id"] for row in aggregates])
    assert comparison["strictly_comparable"]
    assert comparison["comparison_kind"] == "controlled_treatment"
    assert comparison["treatment_fields"] == ["optimizer", "warmup_steps"]
    optimizer = next(
        item for item in comparison["dimensions"] if item["field"] == "optimizer"
    )
    assert optimizer["role"] == "treatment"
    assert not optimizer["match"]
