from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from asterlm.config import DataConfig, TrainConfig
from asterlm.data.clean_manifest import build_clean_corpus_manifest
from asterlm.experiments import MODAL_PROMOTION_GATES, REQUIRED_FINAL_RUN_GATES
from asterlm.training.contracts import (
    TrainingContractError,
    canonical_data_config_sha256,
    validate_clean_manifest,
    validate_training_contract,
)


def _write_clean_source(path: Path, text: str) -> None:
    path.mkdir(parents=True)
    (path / "clean-00000.jsonl").write_text(
        json.dumps({"text": text}) + "\n", encoding="utf-8"
    )
    (path / "cleaning_report.json").write_text(
        json.dumps({"estimated_tokens": len(text) // 4}), encoding="utf-8"
    )


def _sealed_data(tmp_path: Path) -> tuple[Path, DataConfig]:
    train = tmp_path / "clean" / "train"
    validation = tmp_path / "clean" / "validation" / "train"
    _write_clean_source(train, "decision grade training text")
    validation.mkdir(parents=True)
    (validation / "clean-00000.jsonl").write_text(
        json.dumps({"text": "held out text"}) + "\n", encoding="utf-8"
    )
    manifest = tmp_path / "clean" / "clean_manifest.json"
    config_path = tmp_path / "clean" / "pretrain_data.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "manifest_path": str(manifest),
                    "validation_role": "architecture_holdout",
                    "sources": [{"path": str(train), "weight": 1.0}],
                    "validation_sources": [
                        {"path": str(validation), "weight": 1.0}
                    ],
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    build_clean_corpus_manifest(
        data_config_path=config_path,
        output_path=manifest,
        repo_root=tmp_path,
        benchmark_decontaminated=True,
        pii_handled=True,
    )
    return config_path, DataConfig.from_yaml(config_path)


def _passed_gates(path: Path) -> None:
    evidence = path.parent / "promotion-evidence"
    evidence.mkdir(exist_ok=True)
    records = {}
    for gate in [*REQUIRED_FINAL_RUN_GATES, *MODAL_PROMOTION_GATES, "energy_and_power"]:
        artifact = evidence / f"{gate}-result.json"
        artifact.write_text(json.dumps({"gate_id": gate, "passed": True}), encoding="utf-8")
        proof = evidence / f"{gate}-proof.json"
        proof.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "gate_id": gate,
                    "status": "passed",
                    "evaluator": {
                        "name": "training-contract-test",
                        "version": "1",
                        "git_commit": "b" * 40,
                    },
                    "experiment_ids": [f"test-{gate}"],
                    "artifacts": [
                        {
                            "path": artifact.relative_to(path.parent).as_posix(),
                            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        records[gate] = {
            "path": proof.relative_to(path.parent).as_posix(),
            "sha256": hashlib.sha256(proof.read_bytes()).hexdigest(),
        }
    payload = {
        "schema_version": 2,
        "repo_root": ".",
        "allowed_statuses": ["not_run", "running", "passed", "failed", "blocked"],
        "gates": [
            {
                "id": gate,
                "required": True,
                "status": "passed",
                "evidence": [records[gate]],
            }
            for gate in REQUIRED_FINAL_RUN_GATES
        ]
        + [
            {
                "id": gate,
                "required": False,
                "status": "passed",
                "evidence": [records[gate]],
            }
            for gate in MODAL_PROMOTION_GATES
        ]
        + [
            {
                "id": "energy_and_power",
                "required": False,
                "status": "passed",
                "evidence": [records["energy_and_power"]],
            }
        ],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _sealed_tokenizer(tmp_path: Path, data: DataConfig) -> tuple[Path, Path]:
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text('{"test":"tokenizer"}', encoding="utf-8")
    manifest = tmp_path / "tokenizer_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "data_config_sha256": canonical_data_config_sha256(data),
                "tokenizer": {
                    "sha256": hashlib.sha256(tokenizer.read_bytes()).hexdigest()
                },
                "fertility": [
                    {"source_index": index, "documents": 1, "tokens": 4}
                    for index, _ in enumerate(data.sources)
                ],
            }
        ),
        encoding="utf-8",
    )
    return tokenizer, manifest


def test_decision_grade_contract_requires_and_validates_sealed_data(tmp_path: Path) -> None:
    _, data = _sealed_data(tmp_path)
    report = validate_training_contract(
        TrainConfig(run_class="decision_grade"),
        data,
        source_provenance=None,
        verify_artifact_hashes=True,
    )
    assert report.protected
    assert report.clean_manifest_sha256

    artifact = Path(data.sources[0].path) / "clean-00000.jsonl"
    artifact.write_text("tampered", encoding="utf-8")
    with pytest.raises(TrainingContractError, match="size changed"):
        validate_clean_manifest(data)


def test_final_contract_is_mechanically_promotion_locked(tmp_path: Path) -> None:
    _, data = _sealed_data(tmp_path)
    tokenizer, tokenizer_manifest = _sealed_tokenizer(tmp_path, data)
    gates = tmp_path / "gates.yaml"
    gates.write_text(
        (Path("configs/experiments/promotion_gates.yaml")).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    train = TrainConfig(
        run_class="final",
        promotion_gates_path=str(gates),
        wandb_project="asterlm-frontier",
        hub_repo_id="owner/private-checkpoints",
        hub_private=True,
        hub_include_optimizer=True,
        hub_fail_on_error=True,
        checkpoint_local_budget_gib=150.0,
        tokenizer_path=str(tokenizer),
        tokenizer_manifest_path=str(tokenizer_manifest),
        num_workers=0,
    )
    with pytest.raises(TrainingContractError, match="promotion-locked"):
        validate_training_contract(
            train, data, source_provenance={"dirty": False}
        )

    _passed_gates(gates)
    report = validate_training_contract(
        train, data, source_provenance={"dirty": False}
    )
    assert report.promotion_ready is True


def test_final_contract_rejects_metrics_only_checkpoint_policy(tmp_path: Path) -> None:
    _, data = _sealed_data(tmp_path)
    gates = tmp_path / "gates.yaml"
    _passed_gates(gates)
    train = TrainConfig(
        run_class="final",
        promotion_gates_path=str(gates),
        checkpoint_policy="none",
        wandb_project="asterlm-frontier",
        hub_repo_id="owner/private-checkpoints",
        hub_private=True,
        hub_include_optimizer=True,
        num_workers=0,
    )
    with pytest.raises(TrainingContractError, match="checkpoint_policy=full"):
        validate_training_contract(train, data, source_provenance={"dirty": False})


def test_shared_dedup_database_removes_cross_source_duplicate(tmp_path: Path) -> None:
    duplicate = "same globally duplicated training document " * 8
    source_a = tmp_path / "raw-a.jsonl"
    source_b = tmp_path / "raw-b.jsonl"
    source_a.write_text(json.dumps({"text": duplicate}) + "\n", encoding="utf-8")
    source_b.write_text(json.dumps({"text": duplicate}) + "\n", encoding="utf-8")
    database = tmp_path / "global.sqlite"

    reports = []
    for name, source in (("a", source_a), ("b", source_b)):
        output = tmp_path / f"clean-{name}"
        subprocess.run(
            [
                sys.executable,
                "scripts/clean_corpus.py",
                "--input",
                str(source),
                "--output",
                str(output),
                "--dedup-db",
                str(database),
                "--min-chars",
                "1",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        reports.append(json.loads((output / "cleaning_report.json").read_text()))

    assert reports[0]["counts"]["train_kept"] == 1
    assert reports[1]["counts"]["exact_duplicate"] == 1
    assert reports[1]["counts"].get("train_kept", 0) == 0
