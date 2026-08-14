from __future__ import annotations

import json
import math
import os
import random
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader

from asterlm.artifacts import atomic_write_json
from asterlm.config import AsterConfig, DataConfig, TrainConfig
from asterlm.data import AsterTokenizer, PackedTokenDataset, SFTPackedDataset
from asterlm.experiments import ExperimentRegistry
from asterlm.model import AsterLM
from asterlm.optim import build_optimizer, learning_rate_multiplier
from asterlm.quantization.loqt import iter_loqt_modules, merge_loqt_modules
from asterlm.source_provenance import assert_current_checkout_source

from .analysis_schema import build_analysis_manifest
from .checkpoint import (
    checkpoint_storage_usage,
    enforce_checkpoint_storage_budget,
    load_checkpoint,
    load_data_state,
    load_model_weights,
    pin_kda_backend_from_checkpoint,
    prune_rolling_checkpoints,
    save_checkpoint,
)
from .contracts import validate_model_backend_contract, validate_training_contract
from .execution import probe_execution_backends, resolve_execution_engine
from .hub import HubRunSync, HubUploadQueue, HubUploadTask
from .metrics import JsonlLogger
from .parameter_policy import apply_parameter_training_policy
from .precision import PrecisionManager
from .telemetry import (
    SystemSampler,
    assert_required_gradient_coverage,
    gradient_diagnostics,
    parameter_diagnostics,
    save_diagnostic_bundle,
    static_system_manifest,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class _NullLogger:
    def log(self, values: dict[str, Any]) -> None:
        del values


def format_evaluation_metrics(values: dict[str, Any]) -> str:
    return ", ".join(
        f"{key}={value:.4f}" if isinstance(value, (int, float)) else f"{key}={value}"
        for key, value in values.items()
    )


class Trainer:
    """Single-GPU, VRAM-first trainer for pretraining and response-only SFT.

    It deliberately supports slow memory-saving modes—CPU activation offload,
    low-bit optimizer state, CPU optimizer offload, and checkpointed vocabulary
    projection—because the target machine has ample patience but only 12 GiB VRAM.
    """

    def __init__(
        self,
        model_config: AsterConfig,
        train_config: TrainConfig,
        data_config: DataConfig,
        mode: Literal["pretrain", "sft"] = "pretrain",
        initial_checkpoint: str | None = None,
    ) -> None:
        self.source_provenance = assert_current_checkout_source()
        self.training_contract = validate_training_contract(
            train_config,
            data_config,
            source_provenance=self.source_provenance,
        )
        checkpoint_source = train_config.resume or initial_checkpoint
        if checkpoint_source:
            pin_kda_backend_from_checkpoint(model_config, checkpoint_source)
        validate_model_backend_contract(model_config, train_config)
        self.model_config = model_config
        self.train_config = train_config
        self.data_config = data_config
        self.mode = mode

        seed_everything(train_config.seed)
        torch.set_float32_matmul_precision(train_config.matmul_precision)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        if train_config.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        self.device = torch.device(train_config.device)
        self.autocast_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[train_config.dtype]
        if self.device.type == "cuda" and train_config.dtype == "float16":
            raise ValueError(
                "AsterLM intentionally does not silently train FP16 without loss scaling. "
                "Use bfloat16 on RTX 4080, or float32 for debugging."
            )

        self.tokenizer = AsterTokenizer(train_config.tokenizer_path)
        if self.tokenizer.vocab_size != model_config.vocab_size:
            raise ValueError(
                f"Tokenizer vocab ({self.tokenizer.vocab_size}) != model vocab ({model_config.vocab_size})"
            )
        if train_config.sequence_length > model_config.max_seq_len:
            raise ValueError("Training sequence length exceeds model max_seq_len")
        if (
            train_config.precision_backend == "transformer_engine_fp8"
            and model_config.linear_backend != "transformer_engine"
        ):
            raise ValueError(
                "FP8 was requested, but model.linear_backend is not transformer_engine; "
                "ordinary nn.Linear modules would remain BF16."
            )

        self.execution = resolve_execution_engine(model_config, train_config, self.device)

        self.model = AsterLM(
            model_config,
            named_initialization_seed=(
                train_config.seed if train_config.deterministic_named_initialization else None
            ),
            moe_implementation=self.execution.plan.moe_implementation,
        )
        # Autocast alone does not reduce persistent FP32 parameter storage. Store CUDA
        # weights in BF16 (or FP32 when explicitly requested) before optimizer creation.
        if self.device.type == "cuda" and self.autocast_dtype != torch.float32:
            self.model = self.model.to(device=self.device, dtype=self.autocast_dtype)
            # Recurrent time constants are numerically sensitive and FLA expects FP32.
            for name, parameter in self.model.named_parameters():
                if name.endswith(("A_log", "dt_bias")):
                    parameter.data = parameter.data.float()
        else:
            self.model = self.model.to(self.device)

        self.parameter_training_policy = apply_parameter_training_policy(
            self.model, train_config.parameter_training_policy
        )
        self.precision = PrecisionManager(train_config, self.device, self.autocast_dtype)
        self.optimizer = build_optimizer(self.model, train_config)
        self.step = 0
        self.tokens_seen = 0
        if train_config.resume:
            self.step, self.tokens_seen = load_checkpoint(
                self.model, self.optimizer, train_config.resume
            )
        elif initial_checkpoint:
            load_model_weights(self.model, initial_checkpoint)
        self.grouped_expert_storage = self.model.pack_grouped_expert_storage()

        self._milestones_remaining = [
            token for token in train_config.milestone_tokens if token > self.tokens_seen
        ]
        self._training_started_monotonic = time.perf_counter()
        self._last_checkpoint_monotonic = self._training_started_monotonic
        self._logical_parameters = self.model.effective_parameter_count()
        self._active_parameters = self.model.active_parameter_count()

        tokens_per_update = (
            train_config.sequence_length
            * train_config.micro_batch_size
            * train_config.gradient_accumulation_steps
        )
        token_limited_steps = (
            math.ceil(train_config.max_tokens / tokens_per_update)
            if train_config.max_tokens is not None
            else train_config.max_steps
        )
        self.schedule_total_steps = min(train_config.max_steps, token_limited_steps)
        if self.schedule_total_steps > 1 and train_config.warmup_steps >= self.schedule_total_steps:
            raise ValueError(
                f"warmup_steps ({train_config.warmup_steps}) must be smaller than the "
                f"effective schedule horizon ({self.schedule_total_steps})"
            )

        if train_config.num_workers != 0:
            # Worker scheduling and prefetched-but-not-consumed batches are not yet
            # represented by the cursor protocol. Exact checkpoints therefore use
            # the main process until worker queues become checkpointable too.
            print(
                f"exact data checkpoints require num_workers=0; "
                f"overriding requested num_workers={train_config.num_workers}"
            )
            train_config.num_workers = 0
            train_config.prefetch_factor = None

        self.train_loader = self._build_loader(validation=False)
        self.validation_loader = (
            self._build_loader(validation=True) if data_config.validation_sources else None
        )
        resume_data_state = load_data_state(train_config.resume) if train_config.resume else None
        if resume_data_state is not None:
            self._restore_training_data_state(resume_data_state)
        elif train_config.resume and self.tokens_seen:
            if os.environ.get("ASTERLM_ALLOW_LEGACY_DATA_REPLAY") != "1":
                raise RuntimeError(
                    "Resume checkpoint has no exact data_state.pt. Set "
                    "ASTERLM_ALLOW_LEGACY_DATA_REPLAY=1 only for a bounded legacy run; "
                    "final/remote training must use checkpointable data cursors."
                )
        self.train_iterator = iter(self.train_loader)
        self.validation_iterator = (
            iter(self.validation_loader) if self.validation_loader is not None else None
        )
        if train_config.resume and self.tokens_seen and resume_data_state is None:
            self._restore_training_data_position()

        self.forward_model = self.execution.prepare_model(self.model)

        self.output = Path(train_config.output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        previous_metrics: dict[str, Any] = {}
        metrics_path = self.output / "metrics.jsonl"
        if metrics_path.is_file():
            try:
                for line in metrics_path.read_text(encoding="utf-8", errors="replace").splitlines()[-200:]:
                    row = json.loads(line)
                    if row.get("tokens_per_second") is not None:
                        previous_metrics = row
            except (OSError, json.JSONDecodeError):
                previous_metrics = {}
        self._wall_clock_offset_seconds = float(
            previous_metrics.get("wall_clock_campaign_seconds", 0.0)
        )
        self._throughput_ema = (
            float(previous_metrics["tokens_per_second_ema"])
            if previous_metrics.get("tokens_per_second_ema") is not None
            else None
        )
        self.logger = (
            JsonlLogger(self.output / "metrics.jsonl")
            if train_config.jsonl_metrics
            else _NullLogger()
        )
        self.system_sampler = SystemSampler(
            self.device,
            min_interval=train_config.system_metrics_interval,
            energy_joules=float(previous_metrics.get("gpu_energy_joules_total", 0.0)),
        )
        self.tensorboard = None
        if train_config.tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.tensorboard = SummaryWriter(log_dir=str(self.output / "tensorboard"))
            except ImportError as exc:
                raise ImportError("tensorboard=true but tensorboard is not installed") from exc
        self.wandb = None
        wandb_resume = False
        if train_config.wandb_project:
            stored_wandb_id = self._stored_wandb_run_id(
                self.output,
                checkpoint_source=checkpoint_source,
            )
            if (
                train_config.wandb_run_id
                and stored_wandb_id
                and train_config.wandb_run_id != stored_wandb_id
            ):
                raise RuntimeError(
                    "Configured W&B run ID does not match the existing experiment identity"
                )
            wandb_resume = stored_wandb_id is not None
            train_config.wandb_run_id = (
                train_config.wandb_run_id or stored_wandb_id or uuid.uuid4().hex
            )
            train_config.wandb_entity = train_config.wandb_entity or os.environ.get("WANDB_ENTITY")

        manifest: dict[str, Any] = {
            "model": model_config.to_dict(),
            "train": train_config.to_dict(),
            "data": data_config.to_dict(),
            "architecture": self.model.architecture_summary(),
            "system": static_system_manifest(self.device),
            "optimizer_partition": getattr(self.optimizer, "partition", None).__dict__,
            "parameter_storage": self._parameter_storage_summary(),
            "parameter_training_policy": self.parameter_training_policy,
            "grouped_expert_storage": self.grouped_expert_storage,
            "loqt_modules": sum(1 for _ in iter_loqt_modules(self.model)),
            "execution_plan": self.execution.plan.to_dict(),
            "training_contract": self.training_contract.to_dict(),
            "execution_backends": {
                name: capability.to_dict()
                for name, capability in probe_execution_backends(self.device).items()
            },
            "source_provenance": self.source_provenance,
        }
        remote_profile_path = os.environ.get("ASTERLM_REMOTE_EXECUTION_PROFILE")
        if remote_profile_path and Path(remote_profile_path).is_file():
            manifest["remote_execution_profile"] = json.loads(
                Path(remote_profile_path).read_text(encoding="utf-8")
            )
        self.registry = ExperimentRegistry.create(
            self.output,
            repo_root=Path(__file__).resolve().parents[3],
            model=model_config.to_dict(),
            train=train_config.to_dict(),
            data=data_config.to_dict(),
            environment=manifest["system"],
            architecture=manifest["architecture"],
            command=[sys.executable, *sys.argv],
            parent_run_id=self._parent_run_id(checkpoint_source),
            hypothesis=os.environ.get("ASTERLM_EXPERIMENT_HYPOTHESIS"),
            stage=mode,
            resume_existing=bool(train_config.resume),
        )
        manifest["run_id"] = self.registry.record["run_id"]
        atomic_write_json(self.output / "run_manifest.json", manifest)
        atomic_write_json(
            self.output / "analysis_manifest.json",
            build_analysis_manifest(train_config),
        )

        if train_config.wandb_project:
            try:
                import wandb

                self.wandb = wandb
                run = wandb.init(
                    project=train_config.wandb_project,
                    entity=train_config.wandb_entity,
                    id=train_config.wandb_run_id,
                    resume="must" if wandb_resume else "allow",
                    name=train_config.wandb_run_name,
                    config={
                        "model": model_config.to_dict(),
                        "train": train_config.to_dict(),
                        "data": data_config.to_dict(),
                    },
                )
                self.registry.set_wandb_identity(
                    entity=train_config.wandb_entity,
                    project=train_config.wandb_project,
                    run_id=str(train_config.wandb_run_id),
                    url=getattr(run, "url", None),
                )
                run.define_metric("tokens_seen")
                run.define_metric("*", step_metric="tokens_seen")
            except ImportError as exc:
                raise ImportError("wandb_project is set, but wandb is not installed") from exc

        self.hub: HubRunSync | None = None
        self.hub_upload_queue: HubUploadQueue | None = None
        hub_repo_id = os.environ.get("ASTERLM_HUB_REPO_ID") or train_config.hub_repo_id
        if hub_repo_id:
            try:
                self.hub = HubRunSync(
                    repo_id=hub_repo_id,
                    private=train_config.hub_private,
                    revision=train_config.hub_revision,
                    include_optimizer=train_config.hub_include_optimizer,
                    storage_guard_bytes=(
                        int(train_config.hub_storage_guard_tb_decimal * 1_000_000_000_000)
                        if train_config.hub_storage_guard_tb_decimal is not None
                        else None
                    ),
                    storage_hard_cap_bytes=(
                        int(train_config.hub_storage_hard_cap_tb_decimal * 1_000_000_000_000)
                        if train_config.hub_storage_hard_cap_tb_decimal is not None
                        else None
                    ),
                )
                print(f"Hugging Face experiment backup enabled: {hub_repo_id}")
                if train_config.hub_async_upload:
                    self.hub_upload_queue = HubUploadQueue(
                        self.hub,
                        max_pending=train_config.hub_max_pending_uploads,
                    )
                    print(
                        "Asynchronous Hub checkpoint transfer enabled: "
                        f"max_pending={train_config.hub_max_pending_uploads}"
                    )
            except Exception as exc:
                if train_config.hub_fail_on_error:
                    raise
                print(f"WARNING: Hugging Face backup initialization failed: {exc}")

    @staticmethod
    def _stored_wandb_run_id(
        output: Path,
        *,
        checkpoint_source: str | None = None,
    ) -> str | None:
        candidates = [output / ExperimentRegistry.filename]
        resolved: Path | None = None
        if checkpoint_source:
            from .checkpoint import resolve_checkpoint

            resolved = resolve_checkpoint(checkpoint_source)
            candidates.extend(
                [
                    resolved / ExperimentRegistry.filename,
                    resolved.parent / ExperimentRegistry.filename,
                ]
            )
        for path in candidates:
            if not path.is_file():
                continue
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            value = (record.get("metrics") or {}).get("wandb_run_id")
            if value:
                return str(value)
        if resolved is not None:
            config_path = resolved / "train_config.yaml"
            if config_path.is_file():
                try:
                    return TrainConfig.from_yaml(config_path).wandb_run_id
                except (OSError, TypeError, ValueError):
                    return None
        return None

    @staticmethod
    def _parent_run_id(checkpoint_source: str | None) -> str | None:
        if not checkpoint_source:
            return None
        checkpoint = Path(checkpoint_source)
        candidates = [checkpoint / "experiment.json", checkpoint.parent / "experiment.json"]
        for path in candidates:
            try:
                return str(json.loads(path.read_text(encoding="utf-8"))["run_id"])
            except Exception:
                continue
        return None

    def _parameter_storage_summary(self) -> dict[str, Any]:
        by_dtype: dict[str, dict[str, int]] = {}
        seen: set[int] = set()
        total_bytes = 0
        for parameter in self.model.parameters():
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            key = str(parameter.dtype).removeprefix("torch.")
            item = by_dtype.setdefault(key, {"parameters": 0, "bytes": 0})
            item["parameters"] += parameter.numel()
            item["bytes"] += parameter.numel() * parameter.element_size()
            total_bytes += parameter.numel() * parameter.element_size()
        buffer_bytes = 0
        buffer_by_dtype: dict[str, int] = {}
        for buffer in self.model.buffers():
            key = str(buffer.dtype).removeprefix("torch.")
            size = buffer.numel() * buffer.element_size()
            buffer_by_dtype[key] = buffer_by_dtype.get(key, 0) + size
            buffer_bytes += size
        return {
            "trainable_by_dtype": by_dtype,
            "trainable_parameter_gib": total_bytes / 2**30,
            "buffers_by_dtype_bytes": buffer_by_dtype,
            "buffer_gib": buffer_bytes / 2**30,
            "total_persistent_model_gib": (total_bytes + buffer_bytes) / 2**30,
        }

    def _build_loader(self, validation: bool) -> DataLoader:
        if self.mode == "pretrain":
            dataset = PackedTokenDataset(
                self.tokenizer,
                self.data_config,
                self.train_config.sequence_length,
                validation=validation,
                ignore_index=self.train_config.ignore_index,
            )
        else:
            dataset = SFTPackedDataset(
                self.tokenizer,
                self.data_config,
                self.train_config.sequence_length,
                validation=validation,
                ignore_index=self.train_config.ignore_index,
            )
        kwargs: dict[str, Any] = {
            "dataset": dataset,
            "batch_size": self.train_config.micro_batch_size,
            "num_workers": self.train_config.num_workers,
            "pin_memory": self.train_config.pin_memory and self.device.type == "cuda",
        }
        if self.train_config.num_workers > 0 and self.train_config.prefetch_factor is not None:
            kwargs["prefetch_factor"] = self.train_config.prefetch_factor
        return DataLoader(**kwargs)

    def _restore_training_data_position(self) -> None:
        """Replay the deterministic local/streamed mixture to the saved batch boundary.

        Checkpoints contain model/optimizer/RNG state and the exact token count. The
        packed iterable itself is reconstructed from its fixed config/seed, then
        advanced without moving batches to the GPU. This is slower than serializing
        every source cursor and packing buffer, but it prevents silently repeating
        data after an interrupted run.
        """
        tokens_per_microbatch = self.train_config.sequence_length * self.train_config.micro_batch_size
        if self.tokens_seen % tokens_per_microbatch:
            raise RuntimeError(
                f"Saved tokens_seen={self.tokens_seen} is not divisible by the configured "
                f"microbatch size ({tokens_per_microbatch} tokens). The resume config does "
                "not match the checkpoint."
            )
        batches = self.tokens_seen // tokens_per_microbatch
        expected = self.step * self.train_config.gradient_accumulation_steps
        if batches != expected:
            raise RuntimeError(
                f"Checkpoint data position is inconsistent: tokens imply {batches} microbatches "
                f"but step/accumulation imply {expected}."
            )
        if batches == 0:
            return
        print(f"restoring packed-data position by replaying {batches:,} consumed microbatches")
        for index in range(batches):
            try:
                next(self.train_iterator)
            except StopIteration:
                self.train_iterator = iter(self.train_loader)
                next(self.train_iterator)
            if (index + 1) % 100_000 == 0:
                print(f"data-position replay: {index + 1:,}/{batches:,} microbatches")

    def _restore_training_data_state(self, state: dict[str, Any]) -> None:
        if state.get("schema_version") != 1:
            raise RuntimeError("Unsupported training data checkpoint schema")
        if (
            int(state.get("step", -1)) != self.step
            or int(state.get("tokens_seen", -1)) != self.tokens_seen
        ):
            raise RuntimeError("Data cursor step/token identity does not match trainer state")
        train_state = state.get("train")
        if not isinstance(train_state, dict):
            raise RuntimeError("Checkpoint is missing the training data cursor")
        load = getattr(self.train_loader.dataset, "load_state_dict", None)
        if not callable(load):
            raise RuntimeError("Training dataset cannot restore its checkpoint cursor")
        load(train_state)
        validation_state = state.get("validation")
        if validation_state is not None:
            if self.validation_loader is None:
                raise RuntimeError(
                    "Checkpoint contains validation state but no validation data is configured"
                )
            validation_load = getattr(self.validation_loader.dataset, "load_state_dict", None)
            if not callable(validation_load):
                raise RuntimeError("Validation dataset cannot restore its checkpoint cursor")
            validation_load(validation_state)

    def _training_data_state(self) -> dict[str, Any]:
        train_state = self.train_loader.dataset.state_dict()
        validation_state = (
            self.validation_loader.dataset.state_dict()
            if self.validation_loader is not None and self.validation_iterator is not None
            else None
        )
        return {
            "schema_version": 1,
            "step": self.step,
            "tokens_seen": self.tokens_seen,
            "train": train_state,
            "validation": validation_state,
        }

    def _data_cursor_metrics(self) -> dict[str, Any]:
        """Flatten replay/position evidence without logging the cursor payload itself."""

        state = self.train_loader.dataset.state_dict()

        def find_mixture(value: Any) -> dict[str, Any] | None:
            if isinstance(value, dict):
                if isinstance(value.get("source_epochs"), list) and isinstance(
                    value.get("sources"), list
                ):
                    return value
                for nested in value.values():
                    found = find_mixture(nested)
                    if found is not None:
                        return found
            return None

        mixture = find_mixture(state)
        if mixture is None:
            return {"data_cursor_observable": 0}
        result: dict[str, Any] = {"data_cursor_observable": 1}
        epochs = [int(value) for value in mixture["source_epochs"]]
        result["data_source_epoch_max"] = max(epochs, default=0)
        result["data_source_epoch_mean"] = sum(epochs) / max(1, len(epochs))
        for index, (epoch, source_state) in enumerate(
            zip(epochs, mixture["sources"], strict=True)
        ):
            source = source_state.get("source") or {}
            raw_name = str(source.get("name") or Path(str(source.get("path") or index)).name)
            name = "".join(char if char.isalnum() else "_" for char in raw_name).strip("_")
            name = name[:64] or str(index)
            result[f"data_epoch_{name}"] = epoch
            if source_state.get("file_index") is not None:
                result[f"data_file_index_{name}"] = int(source_state["file_index"])
            if source_state.get("record_index") is not None:
                result[f"data_record_index_{name}"] = int(source_state["record_index"])
        return result

    def _next_batch(self, validation: bool = False) -> dict[str, torch.Tensor]:
        iterator = self.validation_iterator if validation else self.train_iterator
        if iterator is None:
            raise RuntimeError("No validation iterator configured")
        try:
            batch = next(iterator)
        except StopIteration:
            loader = self.validation_loader if validation else self.train_loader
            assert loader is not None
            iterator = iter(loader)
            if validation:
                self.validation_iterator = iterator
            else:
                self.train_iterator = iterator
            batch = next(iterator)
        return {key: value.to(self.device, non_blocking=True) for key, value in batch.items()}

    def _forward(self, batch: dict[str, torch.Tensor]):
        with self.precision.activation_context():
            with self.precision.forward_context():
                return self.forward_model(
                    **batch,
                    ignore_index=self.train_config.ignore_index,
                    return_logits=False,
                )

    @torch.no_grad()
    def evaluate(self) -> dict[str, Any]:
        if self.validation_iterator is None:
            return {}
        self.model.eval()
        losses: list[float] = []
        main_losses: list[float] = []
        for _ in range(self.train_config.eval_batches):
            batch = self._next_batch(validation=True)
            with self.precision.forward_context():
                output = self.forward_model(
                    **batch,
                    ignore_index=self.train_config.ignore_index,
                    return_logits=False,
                )
            losses.append(float(output.loss.detach()))
            main_losses.append(float(output.main_loss.detach()))
        self.model.train()
        mean_loss = sum(losses) / len(losses)
        mean_main = sum(main_losses) / len(main_losses)
        return {
            "eval_loss": mean_loss,
            "eval_main_loss": mean_main,
            "eval_perplexity": math.exp(min(mean_main, 20.0)),
            "eval_role": self.data_config.validation_role,
        }

    def _log(self, values: dict[str, Any]) -> None:
        payload = {
            "wall_time_unix": time.time(),
            "step": self.step,
            "tokens_seen": self.tokens_seen,
            **values,
        }
        self.logger.log(payload)
        if self.tensorboard is not None:
            for key, value in payload.items():
                if isinstance(value, (int, float)):
                    self.tensorboard.add_scalar(key, value, self.step)
        if self.wandb is not None:
            self.wandb.log(payload, step=self.step)

    def _save(
        self,
        reason: str,
        *,
        permanent: bool = False,
        tag: str | None = None,
    ) -> Path:
        # Reconcile uploads completed since the previous checkpoint. This keeps
        # ephemeral disks bounded without waiting for work still in flight, and
        # surfaces remote failures at a safe optimizer-update boundary.
        self._harvest_hub_uploads()
        should_upload = self.hub is not None and (
            self.train_config.hub_upload_every_save
            or (reason.startswith("milestone-") and self.train_config.hub_upload_milestones)
            or (reason == "complete" and self.train_config.hub_upload_final)
            or (reason == "studio-stop" and self.train_config.hub_upload_on_stop)
        )
        path = save_checkpoint(
            self.train_config.output_dir,
            self.step,
            self.model,
            self.optimizer,
            self.model_config,
            self.train_config,
            self.tokens_seen,
            self.train_config.keep_last_checkpoints,
            tag=tag,
            permanent=permanent,
            reason=reason,
            data_state=self._training_data_state(),
            prune=False,
        )
        self.registry.add_checkpoint(path, reason=reason)
        diagnostic_bundle: Path | None = None
        if self.train_config.save_diagnostic_bundle:
            diagnostic_bundle = save_diagnostic_bundle(
                self.train_config.output_dir,
                reason=reason,
                extra={"step": self.step, "tokens_seen": self.tokens_seen, "checkpoint": str(path)},
            )
            if self.wandb is not None:
                artifact = self.wandb.Artifact(
                    f"{self.output.name}-diagnostics",
                    type="run-diagnostics",
                    metadata={"step": self.step, "tokens_seen": self.tokens_seen, "reason": reason},
                )
                artifact.add_file(str(diagnostic_bundle))
                self.wandb.log_artifact(artifact, aliases=["latest", reason.replace("/", "-")])

        upload_verified = False
        upload_queued = False
        if should_upload and self.hub is not None:
            if self.hub_upload_queue is not None:
                self.hub_upload_queue.enqueue(
                    HubUploadTask(
                        output_dir=Path(self.train_config.output_dir),
                        checkpoint=path,
                        reason=reason,
                        step=self.step,
                        tokens_seen=self.tokens_seen,
                    )
                )
                upload_queued = True
                self._log({"hub_sync_queued": 1, "hub_sync_checkpoint": str(path)})
            else:
                try:
                    result = self.hub.sync(
                        output_dir=self.train_config.output_dir,
                        checkpoint=path,
                        reason=reason,
                        step=self.step,
                        tokens_seen=self.tokens_seen,
                    )
                    self._log({"hub_sync_seconds": result["seconds"], "hub_sync_ok": 1})
                    upload_verified = result.get("status") == "verified"
                except Exception as exc:
                    self._log({"hub_sync_ok": 0, "hub_sync_error": str(exc)})
                    if self.train_config.hub_fail_on_error:
                        raise
                    print(f"WARNING: Hugging Face checkpoint sync failed: {exc}")
        # Never allow a promised remote checkpoint to trigger local deletion until
        # the Hub copy has passed exact size/hash verification. Ordinary local-only
        # rolling saves still use the configured recent-checkpoint ring.
        if not should_upload or upload_verified:
            prune_rolling_checkpoints(
                self.train_config.output_dir,
                keep_last=self.train_config.keep_last_checkpoints,
                pyramid_levels=self.train_config.checkpoint_pyramid_levels,
            )
        budget_result: dict[str, Any] | None = None
        if (
            self.train_config.checkpoint_local_budget_gib is not None
            and not upload_queued
        ):
            budget_result = enforce_checkpoint_storage_budget(
                self.train_config.output_dir,
                max_total_gib=self.train_config.checkpoint_local_budget_gib,
                keep_last=self.train_config.keep_last_checkpoints,
                protected={path} if path.exists() else None,
            )
        storage = checkpoint_storage_usage(self.train_config.output_dir)
        self._log(
            {
                "event": "checkpoint_committed",
                "checkpoint_reason": reason,
                "checkpoint_path": str(path),
                "checkpoint_permanent": int(permanent),
                "checkpoint_hub_verified": int(upload_verified),
                "checkpoint_hub_queued": int(upload_queued),
                "checkpoint_local_bytes": storage["total_bytes"],
                "checkpoint_local_count": storage["checkpoint_count"],
                "checkpoint_budget_bytes": (
                    budget_result["limit_bytes"] if budget_result is not None else None
                ),
                "checkpoint_budget_ok": (
                    int(budget_result["within_budget"]) if budget_result is not None else None
                ),
                "checkpoint_evicted_count": (
                    len(budget_result["removed"]) if budget_result is not None else 0
                ),
            }
        )
        self._last_checkpoint_monotonic = time.perf_counter()
        if budget_result is not None and not budget_result["within_budget"]:
            print(
                "WARNING: local checkpoint budget cannot be met without deleting a "
                "recent or unverified recovery checkpoint"
            )
        return path

    def _process_hub_upload_report(
        self,
        report: dict[str, Any],
        *,
        protected: set[Path],
    ) -> None:
        for result in report["results"]:
            self._log(
                {
                    "hub_sync_seconds": result.get("seconds"),
                    "hub_sync_ok": 1,
                    "hub_sync_checkpoint": result.get("checkpoint"),
                }
            )
        for error in report["errors"]:
            self._log(
                {
                    "hub_sync_ok": 0,
                    "hub_sync_checkpoint": error.get("checkpoint"),
                    "hub_sync_error": error.get("error"),
                }
            )
            checkpoint = error.get("checkpoint")
            if checkpoint:
                protected.add(Path(str(checkpoint)).resolve())
        if report["errors"] and self.train_config.hub_fail_on_error:
            raise RuntimeError(f"Asynchronous Hub upload failed: {report['errors']}")
        prune_rolling_checkpoints(
            self.train_config.output_dir,
            keep_last=self.train_config.keep_last_checkpoints,
            pyramid_levels=self.train_config.checkpoint_pyramid_levels,
            protected=protected,
        )
        if self.train_config.checkpoint_local_budget_gib is not None:
            enforce_checkpoint_storage_budget(
                self.train_config.output_dir,
                max_total_gib=self.train_config.checkpoint_local_budget_gib,
                keep_last=self.train_config.keep_last_checkpoints,
                protected=protected,
            )

    def _harvest_hub_uploads(self) -> None:
        if self.hub_upload_queue is None:
            return
        report = self.hub_upload_queue.collect_completed()
        self._process_hub_upload_report(
            report,
            protected=self.hub_upload_queue.pending_checkpoints(),
        )

    def _flush_hub_uploads(self, *, close: bool = False) -> None:
        if self.hub_upload_queue is None:
            return
        upload_queue = self.hub_upload_queue
        report = upload_queue.drain(close=close)
        protected = upload_queue.pending_checkpoints()
        if close:
            self.hub_upload_queue = None
        Trainer._process_hub_upload_report(self, report, protected=protected)

    def _clear_optimizer_state_for(self, parameters: list[torch.nn.Parameter]) -> None:
        if not self.train_config.loqt_reset_optimizer_state:
            return
        target = getattr(self.optimizer, "optimizer", None)
        if target is None:
            # Hybrid optimizer: clear states in each owned torch optimizer.
            for name in ("muon", "adamw"):
                candidate = getattr(self.optimizer, name, None)
                if candidate is not None:
                    for parameter in parameters:
                        candidate.state.pop(parameter, None)
            return
        for parameter in parameters:
            target.state.pop(parameter, None)

    def _maybe_merge_loqt(self) -> dict[str, float]:
        interval = self.train_config.loqt_merge_interval
        if interval <= 0 or self.step == 0 or self.step % interval:
            return {}
        modules = list(iter_loqt_modules(self.model))
        if not modules:
            return {}
        parameters = [parameter for module in modules for parameter in (module.a, module.b)]
        started = time.perf_counter()
        stats = merge_loqt_modules(self.model, on_cpu=self.train_config.loqt_merge_on_cpu)
        self._clear_optimizer_state_for(parameters)
        return {
            "loqt_merged_modules": float(stats.modules),
            "loqt_effective_weights_merged": float(stats.effective_weights),
            "loqt_adapter_parameters_reset": float(stats.adapter_parameters),
            "loqt_merge_seconds": time.perf_counter() - started,
        }

    def train(self) -> None:
        cfg = self.train_config
        self.registry.mark_running()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        window_start = time.perf_counter()
        window_tokens = 0
        window_data_s = 0.0
        window_forward_s = 0.0
        window_backward_s = 0.0
        window_optimizer_s = 0.0
        window_excluded_s = 0.0
        last_eval_step = -1
        last_eval_metrics: dict[str, float] = {}

        try:
            while self.step < cfg.max_steps:
                if cfg.max_tokens is not None and self.tokens_seen >= cfg.max_tokens:
                    break
                multiplier = learning_rate_multiplier(
                    self.step,
                    cfg.warmup_steps,
                    self.schedule_total_steps,
                    min_ratio=cfg.min_lr_ratio,
                    schedule_type=cfg.schedule_type,
                    decay_fraction=cfg.decay_fraction,
                    decay_shape=cfg.decay_shape,
                )
                self.optimizer.set_lr_multiplier(multiplier)
                # Keep scalar accumulation on-device. Converting every microbatch
                # loss component to a Python float serialized the CPU and GPU up to
                # five times per microbatch and disproportionately hurt short MoE
                # kernels. Values cross to the host only in an already-synchronized
                # logging window.
                accumulated_loss = torch.zeros((), device=self.device, dtype=torch.float32)
                accumulated_main = torch.zeros_like(accumulated_loss)
                accumulated_mtp = torch.zeros_like(accumulated_loss)
                accumulated_router_aux = torch.zeros_like(accumulated_loss)
                accumulated_router_z = torch.zeros_like(accumulated_loss)

                for _ in range(cfg.gradient_accumulation_steps):
                    started = time.perf_counter()
                    batch = self._next_batch()
                    window_data_s += time.perf_counter() - started

                    started = time.perf_counter()
                    output = self._forward(batch)
                    loss = output.loss / cfg.gradient_accumulation_steps
                    window_forward_s += time.perf_counter() - started

                    started = time.perf_counter()
                    self.execution.backward(loss)
                    window_backward_s += time.perf_counter() - started
                    accumulated_loss.add_(output.loss.detach().float())
                    accumulated_main.add_(output.main_loss.detach().float())
                    if output.mtp_loss is not None:
                        accumulated_mtp.add_(output.mtp_loss.detach().float())
                    if output.router_aux_loss is not None:
                        accumulated_router_aux.add_(output.router_aux_loss.detach().float())
                    if output.router_z_loss is not None:
                        accumulated_router_z.add_(output.router_z_loss.detach().float())
                    batch_tokens = batch["input_ids"].numel()
                    self.tokens_seen += batch_tokens
                    window_tokens += batch_tokens

                diagnostics: dict[str, Any] = {}
                if self.step == 0:
                    diagnostics.update(assert_required_gradient_coverage(self.model))
                if (self.step + 1) % cfg.diagnostic_interval == 0:
                    diagnostics = {
                        **gradient_diagnostics(self.model),
                        **parameter_diagnostics(self.model),
                        **self.model.moe_pathway_stats(),
                        **self.model.moe_execution_stats(),
                        **self._data_cursor_metrics(),
                    }
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), cfg.max_grad_norm
                )
                # Do not convert/check each microbatch loss on the host. With high
                # gradient accumulation that serialized every forward pass and left
                # the GPU idle between otherwise independent queued kernels. The
                # gradient check below is the single pre-step safety gate: any NaN or
                # Inf produced by a loss/backward is caught before parameters mutate.
                if not torch.isfinite(grad_norm).all():
                    bad_grads: list[str] = []
                    for name, parameter in self.model.named_parameters():
                        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                            bad_grads.append(name)
                            if len(bad_grads) >= 12:
                                break
                    raise FloatingPointError(
                        f"Non-finite gradients before optimizer.step at step {self.step}; "
                        f"grad_norm={float(grad_norm)}; first_bad_grad_tensors={bad_grads}"
                    )
                started = time.perf_counter()
                collect_optimizer_diagnostics = (
                    (self.step + 1) % cfg.diagnostic_interval == 0
                )
                self.optimizer.set_diagnostics_enabled(collect_optimizer_diagnostics)
                self.execution.optimizer_step(self.optimizer)
                window_optimizer_s += time.perf_counter() - started
                if collect_optimizer_diagnostics:
                    diagnostics.update(self.optimizer.diagnostics())
                # Bias updates stay on-device every step. Converting their summary
                # tensors to Python scalars would otherwise synchronize the GPU four
                # times per optimizer step, so collect them only when we will log.
                moe_balance_stats = self.model.update_moe_router_biases(
                    collect_stats=(self.step + 1) % cfg.log_interval == 0
                )
                clip_stats = {"qk_heads_clipped": 0.0, "qk_max_logit": 0.0}
                if cfg.qk_clip_interval > 0 and (self.step + 1) % cfg.qk_clip_interval == 0:
                    clip_stats = self.model.apply_qk_clip()
                self.optimizer.zero_grad(set_to_none=True)
                self.step += 1
                loqt_stats = self._maybe_merge_loqt()

                if self.step % cfg.log_interval == 0:
                    if self.device.type == "cuda":
                        # Makes wall-clock and phase timings honest for the logged window.
                        torch.cuda.synchronize(self.device)
                    wall_elapsed = max(time.perf_counter() - window_start, 1e-9)
                    elapsed = max(wall_elapsed - window_excluded_s, 1e-9)
                    grad_norm_value = float(grad_norm)
                    grad_clip_coefficient = min(
                        1.0,
                        cfg.max_grad_norm / max(grad_norm_value, 1e-12),
                    )
                    values: dict[str, Any] = {
                        "loss": float(accumulated_loss / cfg.gradient_accumulation_steps),
                        "main_loss": float(accumulated_main / cfg.gradient_accumulation_steps),
                        "mtp_loss": float(accumulated_mtp / cfg.gradient_accumulation_steps),
                        "router_aux_loss": float(
                            accumulated_router_aux / cfg.gradient_accumulation_steps
                        ),
                        "router_z_loss": float(
                            accumulated_router_z / cfg.gradient_accumulation_steps
                        ),
                        # clip_grad_norm_ returns the norm *before* clipping. Keep
                        # the legacy key for old dashboards and add unambiguous
                        # fields for optimizer stability comparisons.
                        "grad_norm_clipped": grad_norm_value,
                        "grad_norm_pre_clip": grad_norm_value,
                        "grad_clip_coefficient": grad_clip_coefficient,
                        "grad_was_clipped": int(grad_clip_coefficient < 1.0),
                        "lr_multiplier": multiplier,
                        "tokens_per_second": window_tokens / elapsed,
                        "window_seconds": elapsed,
                        "window_wall_seconds": wall_elapsed,
                        "window_excluded_seconds": window_excluded_s,
                        "data_wait_seconds": window_data_s,
                        "forward_submit_seconds": window_forward_s,
                        "backward_submit_seconds": window_backward_s,
                        "optimizer_submit_seconds": window_optimizer_s,
                        "effective_batch_tokens": cfg.sequence_length
                        * cfg.micro_batch_size
                        * cfg.gradient_accumulation_steps,
                        "progress_fraction": (
                            self.tokens_seen / cfg.max_tokens if cfg.max_tokens else self.step / cfg.max_steps
                        ),
                        "tokens_per_logical_parameter": self.tokens_seen / self._logical_parameters,
                        "tokens_per_active_parameter": self.tokens_seen / self._active_parameters,
                        "estimated_training_tflops": (
                            window_tokens / elapsed * 6.0 * self._active_parameters / 1e12
                        ),
                        "estimated_cumulative_flops": 6.0 * self._active_parameters * self.tokens_seen,
                        "wall_clock_total_seconds": time.perf_counter() - self._training_started_monotonic,
                        "eta_seconds": (
                            ((cfg.max_tokens - self.tokens_seen) / max(window_tokens / elapsed, 1e-9))
                            if cfg.max_tokens is not None
                            else ((cfg.max_steps - self.step) * elapsed / max(1, cfg.log_interval))
                        ),
                        **clip_stats,
                        **moe_balance_stats,
                        **loqt_stats,
                        **diagnostics,
                        **self.system_sampler.sample(force=True),
                    }
                    self._throughput_ema = (
                        values["tokens_per_second"]
                        if self._throughput_ema is None
                        else 0.9 * self._throughput_ema + 0.1 * values["tokens_per_second"]
                    )
                    values["tokens_per_second_ema"] = self._throughput_ema
                    values["wall_clock_campaign_seconds"] = (
                        self._wall_clock_offset_seconds + values["wall_clock_total_seconds"]
                    )
                    values["eta_smoothed_seconds"] = (
                        (cfg.max_tokens - self.tokens_seen) / max(self._throughput_ema, 1e-9)
                        if cfg.max_tokens is not None
                        else values["eta_seconds"]
                    )
                    self._log(values)
                    self.registry.update_progress(
                        self.tokens_seen,
                        throughput_tokens_s=values.get("tokens_per_second"),
                        peak_vram_gib=values.get("cuda_peak_allocated_gb"),
                        energy_tokens_per_joule=(
                            values.get("tokens_per_second") / values.get("gpu_power_w")
                            if values.get("tokens_per_second") and values.get("gpu_power_w")
                            else None
                        ),
                    )
                    print(
                        f"step={self.step:,} tokens={self.tokens_seen:,} "
                        f"loss={values['loss']:.4f} tok/s={values['tokens_per_second']:.0f} "
                        f"vram={values.get('cuda_peak_allocated_gb', 0):.2f}GiB"
                    )
                    if self.device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(self.device)
                    window_start = time.perf_counter()
                    window_tokens = 0
                    window_data_s = window_forward_s = window_backward_s = window_optimizer_s = 0.0
                    window_excluded_s = 0.0

                if self.validation_iterator is not None and self.step % cfg.eval_interval == 0:
                    excluded_started = time.perf_counter()
                    metrics = self.evaluate()
                    last_eval_step = self.step
                    last_eval_metrics = metrics
                    self._log(metrics)
                    print("evaluation:", format_evaluation_metrics(metrics))
                    window_excluded_s += time.perf_counter() - excluded_started

                checkpoint_due_by_step = self.step % cfg.save_interval == 0
                checkpoint_due_by_time = (
                    cfg.checkpoint_interval_minutes is not None
                    and time.perf_counter() - self._last_checkpoint_monotonic
                    >= cfg.checkpoint_interval_minutes * 60.0
                )
                if cfg.checkpoint_policy == "full" and (
                    checkpoint_due_by_step or checkpoint_due_by_time
                ):
                    excluded_started = time.perf_counter()
                    trigger = "step" if checkpoint_due_by_step else "wall-clock"
                    path = self._save(f"periodic-{trigger}")
                    print(f"saved {path}")
                    window_excluded_s += time.perf_counter() - excluded_started

                while self._milestones_remaining and self.tokens_seen >= self._milestones_remaining[0]:
                    excluded_started = time.perf_counter()
                    milestone = self._milestones_remaining.pop(0)
                    milestone_metrics: dict[str, Any] = {
                        "event": "token_milestone",
                        "milestone_tokens": milestone,
                        "milestone_overshoot_tokens": self.tokens_seen - milestone,
                    }
                    if cfg.milestone_eval and self.validation_iterator is not None:
                        if last_eval_step == self.step:
                            milestone_metrics.update(last_eval_metrics)
                        else:
                            last_eval_metrics = self.evaluate()
                            last_eval_step = self.step
                            milestone_metrics.update(last_eval_metrics)
                    self._log(milestone_metrics)
                    if cfg.checkpoint_policy == "full":
                        path = self._save(
                            f"milestone-{milestone}",
                            permanent=True,
                            tag=f"tok-{milestone}",
                        )
                        print(f"permanent token milestone saved: {path}")
                    else:
                        print(
                            f"token milestone recorded without checkpoint: {milestone:,} "
                            f"(checkpoint_policy={cfg.checkpoint_policy})"
                        )
                    window_excluded_s += time.perf_counter() - excluded_started

            path = None
            if cfg.checkpoint_policy in {"full", "final_only"}:
                path = self._save("complete", permanent=True, tag="final")
            self._flush_hub_uploads(close=True)
            self.registry.finish("ok", tokens_seen=self.tokens_seen)
            if path is None:
                print("training complete; metrics-only run saved no model checkpoint")
            else:
                print(f"training complete; final checkpoint: {path}")
        except BaseException as exc:
            declared_status = getattr(exc, "asterlm_status", None)
            if declared_status:
                status = str(declared_status)
            elif isinstance(exc, KeyboardInterrupt):
                status = "interrupted_user"
            elif isinstance(exc, torch.cuda.OutOfMemoryError):
                status = "failed_oom"
            elif isinstance(exc, FloatingPointError):
                status = "failed_nan"
            else:
                status = "failed_kernel"
            self.registry.finish(status, tokens_seen=self.tokens_seen, reason=str(exc))
            if cfg.save_diagnostic_bundle:
                bundle = save_diagnostic_bundle(
                    cfg.output_dir,
                    reason="failure",
                    extra={
                        "step": self.step,
                        "tokens_seen": self.tokens_seen,
                        "exception_type": type(exc).__name__,
                        "exception": str(exc),
                        "system": self.system_sampler.sample(force=True),
                    },
                )
                print(f"saved failure diagnostic bundle: {bundle}")
                if self.wandb is not None:
                    artifact = self.wandb.Artifact(
                        f"{self.output.name}-failures",
                        type="failure-diagnostics",
                        metadata={"step": self.step, "tokens_seen": self.tokens_seen},
                    )
                    artifact.add_file(str(bundle))
                    self.wandb.log_artifact(artifact, aliases=["latest"])
            raise
        finally:
            if self.hub_upload_queue is not None:
                self._flush_hub_uploads(close=True)
            if self.tensorboard is not None:
                self.tensorboard.flush()
                self.tensorboard.close()
            if self.wandb is not None:
                self.wandb.finish()
