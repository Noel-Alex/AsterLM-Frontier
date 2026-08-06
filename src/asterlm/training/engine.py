from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader

from asterlm.config import AsterConfig, DataConfig, TrainConfig
from asterlm.data import AsterTokenizer, PackedTokenDataset, SFTPackedDataset
from asterlm.model import AsterLM
from asterlm.quantization.loqt import iter_loqt_modules, merge_loqt_modules
from asterlm.optim import build_optimizer, learning_rate_multiplier
from .checkpoint import (
    load_checkpoint,
    load_model_weights,
    pin_kda_backend_from_checkpoint,
    save_checkpoint,
)
from .metrics import JsonlLogger
from .precision import PrecisionManager
from .telemetry import (
    SystemSampler,
    gradient_diagnostics,
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
        checkpoint_source = train_config.resume or initial_checkpoint
        if checkpoint_source:
            pin_kda_backend_from_checkpoint(model_config, checkpoint_source)
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

        self.model = AsterLM(model_config)
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

        if train_config.resume and train_config.num_workers != 0:
            # Worker process scheduling/prefetch state is not represented in our
            # checkpoints. Force a single-process deterministic stream so replaying
            # consumed microbatches restores the exact packed-data position.
            print(
                f"resume requested with num_workers={train_config.num_workers}; "
                "forcing num_workers=0 for deterministic data-position recovery"
            )
            train_config.num_workers = 0
            train_config.prefetch_factor = None

        self.train_loader = self._build_loader(validation=False)
        self.validation_loader = (
            self._build_loader(validation=True) if data_config.validation_sources else None
        )
        self.train_iterator = iter(self.train_loader)
        self.validation_iterator = (
            iter(self.validation_loader) if self.validation_loader is not None else None
        )
        if train_config.resume and self.tokens_seen:
            self._restore_training_data_position()

        self.forward_model = self.model
        if train_config.compile:
            if model_config.linear_backend == "transformer_engine":
                raise ValueError(
                    "Compile and Transformer Engine should be benchmarked separately first; "
                    "the default matrix deliberately forbids stacking unvalidated compilers."
                )
            self.forward_model = torch.compile(
                self.model, mode=train_config.compile_mode, dynamic=False
            )

        self.output = Path(train_config.output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.logger = (
            JsonlLogger(self.output / "metrics.jsonl")
            if train_config.jsonl_metrics
            else _NullLogger()
        )
        self.system_sampler = SystemSampler(
            self.device, min_interval=train_config.system_metrics_interval
        )
        self.tensorboard = None
        if train_config.tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.tensorboard = SummaryWriter(log_dir=str(self.output / "tensorboard"))
            except ImportError as exc:
                raise ImportError("tensorboard=true but tensorboard is not installed") from exc
        self.wandb = None
        if train_config.wandb_project:
            try:
                import wandb

                self.wandb = wandb
                wandb.init(
                    project=train_config.wandb_project,
                    name=train_config.wandb_run_name,
                    config={
                        "model": model_config.to_dict(),
                        "train": train_config.to_dict(),
                        "data": data_config.to_dict(),
                    },
                )
            except ImportError as exc:
                raise ImportError("wandb_project is set, but wandb is not installed") from exc

        manifest: dict[str, Any] = {
            "model": model_config.to_dict(),
            "train": train_config.to_dict(),
            "data": data_config.to_dict(),
            "architecture": self.model.architecture_summary(),
            "system": static_system_manifest(self.device),
            "optimizer_partition": getattr(self.optimizer, "partition", None).__dict__,
            "parameter_storage": self._parameter_storage_summary(),
            "loqt_modules": sum(1 for _ in iter_loqt_modules(self.model)),
        }
        (self.output / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str), encoding="utf-8"
        )

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
    def evaluate(self) -> dict[str, float]:
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

    def _save(self, reason: str) -> Path:
        path = save_checkpoint(
            self.train_config.output_dir,
            self.step,
            self.model,
            self.optimizer,
            self.model_config,
            self.train_config,
            self.tokens_seen,
            self.train_config.keep_last_checkpoints,
        )
        if self.train_config.save_diagnostic_bundle:
            save_diagnostic_bundle(
                self.train_config.output_dir,
                reason=reason,
                extra={"step": self.step, "tokens_seen": self.tokens_seen, "checkpoint": str(path)},
            )
        return path

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
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        window_start = time.perf_counter()
        window_tokens = 0
        window_data_s = 0.0
        window_forward_s = 0.0
        window_backward_s = 0.0
        window_optimizer_s = 0.0

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
                accumulated_loss = 0.0
                accumulated_main = 0.0
                accumulated_mtp = 0.0
                accumulated_router_aux = 0.0
                accumulated_router_z = 0.0

                for _ in range(cfg.gradient_accumulation_steps):
                    started = time.perf_counter()
                    batch = self._next_batch()
                    window_data_s += time.perf_counter() - started

                    started = time.perf_counter()
                    output = self._forward(batch)
                    loss = output.loss / cfg.gradient_accumulation_steps
                    window_forward_s += time.perf_counter() - started
                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            f"Non-finite loss at step {self.step}: {float(loss)}"
                        )

                    started = time.perf_counter()
                    loss.backward()
                    window_backward_s += time.perf_counter() - started
                    accumulated_loss += float(output.loss.detach())
                    accumulated_main += float(output.main_loss.detach())
                    if output.mtp_loss is not None:
                        accumulated_mtp += float(output.mtp_loss.detach())
                    if output.router_aux_loss is not None:
                        accumulated_router_aux += float(output.router_aux_loss.detach())
                    if output.router_z_loss is not None:
                        accumulated_router_z += float(output.router_z_loss.detach())
                    batch_tokens = batch["input_ids"].numel()
                    self.tokens_seen += batch_tokens
                    window_tokens += batch_tokens

                diagnostics: dict[str, Any] = {}
                if (self.step + 1) % cfg.diagnostic_interval == 0:
                    diagnostics = gradient_diagnostics(self.model)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), cfg.max_grad_norm
                )
                started = time.perf_counter()
                self.optimizer.step()
                window_optimizer_s += time.perf_counter() - started
                moe_balance_stats = self.model.update_moe_router_biases()
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
                    elapsed = max(time.perf_counter() - window_start, 1e-9)
                    values: dict[str, Any] = {
                        "loss": accumulated_loss / cfg.gradient_accumulation_steps,
                        "main_loss": accumulated_main / cfg.gradient_accumulation_steps,
                        "mtp_loss": accumulated_mtp / cfg.gradient_accumulation_steps,
                        "router_aux_loss": accumulated_router_aux / cfg.gradient_accumulation_steps,
                        "router_z_loss": accumulated_router_z / cfg.gradient_accumulation_steps,
                        "grad_norm_clipped": float(grad_norm),
                        "lr_multiplier": multiplier,
                        "tokens_per_second": window_tokens / elapsed,
                        "window_seconds": elapsed,
                        "data_wait_seconds": window_data_s,
                        "forward_submit_seconds": window_forward_s,
                        "backward_submit_seconds": window_backward_s,
                        "optimizer_submit_seconds": window_optimizer_s,
                        "effective_batch_tokens": cfg.sequence_length
                        * cfg.micro_batch_size
                        * cfg.gradient_accumulation_steps,
                        **clip_stats,
                        **moe_balance_stats,
                        **loqt_stats,
                        **diagnostics,
                        **self.system_sampler.sample(force=True),
                    }
                    self._log(values)
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

                if self.validation_iterator is not None and self.step % cfg.eval_interval == 0:
                    metrics = self.evaluate()
                    self._log(metrics)
                    print("evaluation:", ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

                if self.step % cfg.save_interval == 0:
                    path = self._save("periodic")
                    print(f"saved {path}")

            path = self._save("complete")
            print(f"training complete; final checkpoint: {path}")
        except BaseException as exc:
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
            raise
        finally:
            if self.tensorboard is not None:
                self.tensorboard.flush()
                self.tensorboard.close()
            if self.wandb is not None:
                self.wandb.finish()
