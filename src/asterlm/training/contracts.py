from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from asterlm.artifacts import sha256_file
from asterlm.config import DataConfig, TrainConfig
from asterlm.experiments import evaluate_promotion_gates

DECISION_VALIDATION_ROLES = {
    "optimization_validation",
    "architecture_holdout",
    "benchmark_holdout",
    "long_context_holdout",
}
REQUIRED_CLEANING_FLAGS = {
    "cleaned",
    "exact_deduplicated",
    "near_deduplicated",
    "cross_source_deduplicated",
    "benchmark_decontaminated",
    "validation_split_disjoint",
    "pii_handled",
}


class TrainingContractError(RuntimeError):
    """Raised before model allocation when a protected training run is unsafe."""


@dataclass(frozen=True, slots=True)
class TrainingContractReport:
    run_class: str
    protected: bool
    clean_manifest_path: str | None
    clean_manifest_sha256: str | None
    promotion_gates_path: str | None
    promotion_ready: bool | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_class": self.run_class,
            "protected": self.protected,
            "clean_manifest_path": self.clean_manifest_path,
            "clean_manifest_sha256": self.clean_manifest_sha256,
            "promotion_gates_path": self.promotion_gates_path,
            "promotion_ready": self.promotion_ready,
        }


def canonical_data_config_sha256(config: DataConfig) -> str:
    payload = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _manifest_resolved(raw: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    hint = Path(str(payload.get("path_base_hint", ""))).expanduser()
    base = hint if hint.is_dir() else Path.cwd()
    return (base / path).resolve()


def validate_clean_manifest(
    config: DataConfig,
    *,
    verify_artifact_hashes: bool = False,
) -> tuple[Path, dict[str, Any]]:
    if not config.manifest_path:
        raise TrainingContractError(
            "Protected training requires data.manifest_path from the clean-corpus pipeline"
        )
    manifest_path = _resolved(config.manifest_path)
    if not manifest_path.is_file():
        raise TrainingContractError(f"Clean-corpus manifest does not exist: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingContractError(f"Cannot read clean-corpus manifest: {exc}") from exc
    if payload.get("schema_version") != 1 or payload.get("status") != "complete":
        raise TrainingContractError("Clean-corpus manifest must be schema_version=1 and status=complete")

    pipeline = payload.get("pipeline")
    if not isinstance(pipeline, dict):
        raise TrainingContractError("Clean-corpus manifest has no pipeline declaration")
    false_flags = sorted(flag for flag in REQUIRED_CLEANING_FLAGS if pipeline.get(flag) is not True)
    if false_flags:
        raise TrainingContractError(
            "Clean-corpus manifest is not decision-grade; missing guarantees: "
            + ", ".join(false_flags)
        )

    expected_config_sha = canonical_data_config_sha256(config)
    if payload.get("data_config_sha256") != expected_config_sha:
        raise TrainingContractError(
            "Data config does not match the immutable clean-corpus manifest"
        )
    expected_train = {_resolved(source.path) for source in config.sources}
    expected_validation = {_resolved(source.path) for source in config.validation_sources}
    manifest_train = {
        _manifest_resolved(path, payload) for path in payload.get("train_paths", [])
    }
    manifest_validation = {
        _manifest_resolved(path, payload) for path in payload.get("validation_paths", [])
    }
    if expected_train != manifest_train or expected_validation != manifest_validation:
        raise TrainingContractError("Configured train/validation paths differ from the manifest")

    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise TrainingContractError("Clean-corpus manifest contains no hashed artifacts")
    for raw in artifacts:
        if not isinstance(raw, dict) or not raw.get("path") or not raw.get("sha256"):
            raise TrainingContractError("Malformed artifact record in clean-corpus manifest")
        artifact = _manifest_resolved(str(raw["path"]), payload)
        if not artifact.is_file():
            raise TrainingContractError(f"Clean-corpus artifact is missing: {artifact}")
        if artifact.stat().st_size != int(raw.get("size_bytes", -1)):
            raise TrainingContractError(f"Clean-corpus artifact size changed: {artifact}")
        if verify_artifact_hashes and sha256_file(artifact) != str(raw["sha256"]):
            raise TrainingContractError(f"Clean-corpus artifact hash changed: {artifact}")
    return manifest_path, payload


def validate_training_contract(
    train: TrainConfig,
    data: DataConfig,
    *,
    source_provenance: dict[str, Any] | None,
    verify_artifact_hashes: bool = False,
) -> TrainingContractReport:
    if train.run_class == "exploratory":
        return TrainingContractReport(
            run_class=train.run_class,
            protected=False,
            clean_manifest_path=None,
            clean_manifest_sha256=None,
            promotion_gates_path=None,
            promotion_ready=None,
        )

    if data.validation_role not in DECISION_VALIDATION_ROLES:
        raise TrainingContractError(
            f"{train.run_class} runs require an explicit decision-grade validation_role; "
            f"got {data.validation_role!r}"
        )
    manifest_path, _ = validate_clean_manifest(
        data, verify_artifact_hashes=verify_artifact_hashes
    )
    promotion_path: Path | None = None
    promotion_ready: bool | None = None

    if train.run_class == "final":
        if train.checkpoint_policy != "full":
            raise TrainingContractError(
                "Final training requires checkpoint_policy=full for crash recovery and milestones"
            )
        if train.activation_offload:
            raise TrainingContractError(
                "Final pretraining forbids CPU activation offload; use segmented GPU "
                "recomputation so PCIe transfer cannot become the training bottleneck"
            )
        if train.optimizer == "torchao_cpu_offload_adamw":
            raise TrainingContractError(
                "Final pretraining forbids CPU-offloaded optimizer state"
            )
        if train.loqt_merge_interval > 0 and train.loqt_merge_on_cpu:
            raise TrainingContractError(
                "Final pretraining forbids CPU LoQT merge work; select an all-GPU recipe"
            )
        promotion_path = _resolved(train.promotion_gates_path)
        decision = evaluate_promotion_gates(promotion_path)
        promotion_ready = decision.ready
        if not decision.ready:
            preview = ", ".join(decision.blocking_gate_ids[:8])
            remainder = len(decision.blocking_gate_ids) - 8
            if remainder > 0:
                preview += f", and {remainder} more"
            raise TrainingContractError(f"Final training is promotion-locked by: {preview}")
        if source_provenance is None or source_provenance.get("dirty"):
            raise TrainingContractError("Final training requires a clean, source-pinned Git checkout")
        if not train.wandb_project:
            raise TrainingContractError("Final training requires a durable W&B project")
        if not train.hub_repo_id:
            raise TrainingContractError("Final training requires a private Hugging Face checkpoint repo")
        if not train.hub_private or not train.hub_include_optimizer:
            raise TrainingContractError(
                "Final training requires private Hub uploads with full optimizer/RNG resume state"
            )
        if not train.hub_fail_on_error:
            raise TrainingContractError(
                "Final training requires hub_fail_on_error=true so a milestone is never "
                "reported durable after a failed upload"
            )
        if (
            train.hub_storage_guard_tb_decimal is None
            or train.hub_storage_hard_cap_tb_decimal is None
        ):
            raise TrainingContractError(
                "Final training requires explicit Hub storage guard and hard-cap limits"
            )
        if train.checkpoint_local_budget_gib is None:
            raise TrainingContractError(
                "Final training requires an explicit checkpoint_local_budget_gib"
            )
        if not train.tokenizer_manifest_path:
            raise TrainingContractError(
                "Final training requires tokenizer_manifest_path from the sealed tokenizer build"
            )
        tokenizer_path = _resolved(train.tokenizer_path)
        tokenizer_manifest_path = _resolved(train.tokenizer_manifest_path)
        if not tokenizer_path.is_file() or not tokenizer_manifest_path.is_file():
            raise TrainingContractError(
                "Final training requires both the tokenizer and its seal manifest"
            )
        try:
            tokenizer_manifest = json.loads(
                tokenizer_manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise TrainingContractError(f"Cannot read tokenizer seal manifest: {exc}") from exc
        tokenizer_record = tokenizer_manifest.get("tokenizer") or {}
        if (
            tokenizer_manifest.get("schema_version") != 1
            or tokenizer_manifest.get("status") != "complete"
            or tokenizer_record.get("sha256") != sha256_file(tokenizer_path)
        ):
            raise TrainingContractError(
                "Tokenizer seal is incomplete or does not match tokenizer_path"
            )
        if tokenizer_manifest.get("data_config_sha256") != canonical_data_config_sha256(data):
            raise TrainingContractError(
                "Tokenizer was not trained and measured against this immutable clean data config"
            )
        fertility = tokenizer_manifest.get("fertility")
        if not isinstance(fertility, list) or len(fertility) < len(data.sources):
            raise TrainingContractError(
                "Tokenizer seal lacks source-level fertility measurements"
            )
        if train.num_workers != 0:
            raise TrainingContractError(
                "Final training requires num_workers=0 until worker-prefetch queues are checkpointable"
            )
        if not train.jsonl_metrics:
            raise TrainingContractError("Final training cannot disable the local append-only metrics log")

    return TrainingContractReport(
        run_class=train.run_class,
        protected=True,
        clean_manifest_path=str(manifest_path),
        clean_manifest_sha256=sha256_file(manifest_path),
        promotion_gates_path=str(promotion_path) if promotion_path else None,
        promotion_ready=promotion_ready,
    )
