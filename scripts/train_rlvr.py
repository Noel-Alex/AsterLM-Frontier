#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from asterlm.config import AsterConfig, TrainConfig
from asterlm.data import AsterTokenizer
from asterlm.model import AsterLM
from asterlm.optim import build_optimizer, learning_rate_multiplier
from asterlm.quantization.loqt import iter_loqt_modules, merge_loqt_modules
from asterlm.reasoning import RLVRConfig
from asterlm.reasoning.io import iter_json_records
from asterlm.reasoning.losses import rlvr_policy_loss, selected_token_logprobs_from_hidden
from asterlm.reasoning.training import build_rl_sequence, pad_rl_batch
from asterlm.training.checkpoint import load_checkpoint, load_model_weights, pin_kda_backend_from_checkpoint, save_checkpoint
from asterlm.training.metrics import JsonlLogger
from asterlm.training.precision import PrecisionManager
from asterlm.training.telemetry import SystemSampler, gradient_diagnostics, save_diagnostic_bundle


def model_to_training_dtype(model: AsterLM, device: torch.device, dtype: torch.dtype) -> AsterLM:
    if device.type == "cuda" and dtype != torch.float32:
        model = model.to(device=device, dtype=dtype)
        for name, parameter in model.named_parameters():
            if name.endswith(("A_log", "dt_bias")):
                parameter.data = parameter.data.float()
        return model
    return model.to(device)


def clear_optimizer_state(optimizer, parameters: list[torch.nn.Parameter]) -> None:
    target = getattr(optimizer, "optimizer", None)
    if target is not None:
        for parameter in parameters:
            target.state.pop(parameter, None)
        return
    for name in ("muon", "adamw"):
        candidate = getattr(optimizer, name, None)
        if candidate is not None:
            for parameter in parameters:
                candidate.state.pop(parameter, None)


def main() -> None:
    parser = argparse.ArgumentParser(description="VRAM-first GSPO/DAPO/GRPO reasoning update")
    parser.add_argument("--model", required=True)
    parser.add_argument("--train", default="configs/train/reasoning_rl_laptop.yaml")
    parser.add_argument("--reasoning", default="configs/reasoning/rlvr_laptop_gspo.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--resume-optimizer", action="store_true")
    args = parser.parse_args()

    model_config = AsterConfig.from_yaml(args.model)
    pin_kda_backend_from_checkpoint(model_config, args.checkpoint)
    train_config = TrainConfig.from_yaml(args.train)
    rl_config = RLVRConfig.from_yaml(args.reasoning)
    random.seed(rl_config.seed + args.iteration)
    torch.manual_seed(rl_config.seed + args.iteration)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rl_config.seed + args.iteration)
    device = torch.device(train_config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[train_config.dtype]
    if device.type == "cuda" and dtype == torch.float16:
        raise ValueError("Use BF16 rather than unscaled FP16")

    tokenizer = AsterTokenizer(rl_config.tokenizer_path)
    pad_id = tokenizer.token_to_id("<|pad|>")
    model = AsterLM(model_config)
    model = model_to_training_dtype(model, device, dtype)
    precision = PrecisionManager(train_config, device, dtype)
    optimizer = build_optimizer(model, train_config)
    if args.resume_optimizer and (Path(args.checkpoint) / "trainer_state.pt").exists():
        base_step, tokens_seen = load_checkpoint(model, optimizer, args.checkpoint)
    else:
        load_model_weights(model, args.checkpoint)
        base_step, tokens_seen = 0, 0

    raw_records = [record for record in iter_json_records(args.rollouts) if record.get("usable_group", True)]
    if not raw_records:
        raise ValueError("No usable rollout groups. Increase rollout diversity or disable dynamic_sampling for diagnosis.")
    if rl_config.max_groups_per_update > 0:
        allowed_groups = sorted({int(record["group_id"]) for record in raw_records})[: rl_config.max_groups_per_update]
        raw_records = [record for record in raw_records if int(record["group_id"]) in set(allowed_groups)]
    sequences = [build_rl_sequence(record) for record in raw_records]
    if rl_config.kl_beta > 0 and any(sequence.reference_token_logps is None for sequence in sequences):
        raise ValueError("kl_beta > 0 requires score_rl_reference.py output")

    output = Path(rl_config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(output / "rlvr_metrics.jsonl")
    sampler = SystemSampler(device, min_interval=train_config.system_metrics_interval)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    microbatches_per_epoch = math.ceil(len(sequences) / rl_config.micro_batch_size)
    # A short accumulation window is flushed at every epoch boundary, so count each
    # epoch independently rather than combining partial windows across epochs.
    update_steps = (
        math.ceil(microbatches_per_epoch / rl_config.gradient_accumulation_steps)
        * rl_config.update_epochs
    )
    started = time.perf_counter()
    global_update = 0
    running = {"loss": 0.0, "policy": 0.0, "kl": 0.0, "router": 0.0, "clip": 0.0, "ratio": 0.0, "approx_kl": 0.0}
    micro_count = 0
    micro_since_step = 0

    try:
        for epoch in range(rl_config.update_epochs):
            random.shuffle(sequences)
            for start in range(0, len(sequences), rl_config.micro_batch_size):
                chunk = sequences[start : start + rl_config.micro_batch_size]
                batch = pad_rl_batch(chunk, pad_id)
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                targets = batch["targets"].to(device, non_blocking=True)
                mask = batch["completion_mask"].to(device, non_blocking=True)
                old_logps = batch["old_token_logps"].to(device, non_blocking=True)
                reference = batch["reference_token_logps"]
                if reference is not None:
                    reference = reference.to(device, non_blocking=True)
                advantages = batch["advantages"].to(device, non_blocking=True)
                with precision.activation_context(), precision.forward_context():
                    model_output = model(input_ids, return_logits=False, return_hidden=True)
                    hidden_states = model_output.hidden_states
                    if hidden_states is None:
                        raise RuntimeError("RLVR requires hidden states")
                    current = selected_token_logprobs_from_hidden(
                        model,
                        hidden_states,
                        targets,
                        mask,
                        chunk_size=model_config.lm_loss_chunk_size,
                        checkpoint_chunks=train_config.gradient_checkpointing,
                    )
                    loss_output = rlvr_policy_loss(
                        current,
                        old_logps,
                        mask,
                        advantages,
                        algorithm=rl_config.algorithm,
                        clip_low=rl_config.clip_low,
                        clip_high=rl_config.clip_high,
                        ratio_log_clip=rl_config.ratio_log_clip,
                        reference_token_logps=reference,
                        kl_beta=rl_config.kl_beta,
                        entropy_bonus=rl_config.entropy_bonus,
                    )
                    router_regularization = torch.zeros((), device=device)
                    if model_output.router_aux_loss is not None:
                        router_regularization = router_regularization + (
                            model_config.moe_aux_loss_weight * model_output.router_aux_loss
                        )
                    if model_output.router_z_loss is not None:
                        router_regularization = router_regularization + (
                            model_config.moe_router_z_loss_weight * model_output.router_z_loss
                        )
                    objective = loss_output.loss + router_regularization
                    loss = objective / rl_config.gradient_accumulation_steps
                loss.backward()
                micro_count += 1
                micro_since_step += 1
                tokens_seen += int(mask.sum())
                running["loss"] += float(objective.detach())
                running["policy"] += float(loss_output.policy_loss.detach())
                running["kl"] += float(loss_output.kl_loss.detach())
                running["router"] += float(router_regularization.detach())
                running["clip"] += float(loss_output.clip_fraction.detach())
                running["ratio"] += float(loss_output.mean_ratio.detach())
                running["approx_kl"] += float(loss_output.approx_kl.detach())

                end_of_epoch = start + len(chunk) >= len(sequences)
                end_accum = micro_since_step >= rl_config.gradient_accumulation_steps or end_of_epoch
                if not end_accum:
                    continue
                # Each microbatch was divided by the configured accumulation count.
                # Restore the correct average when a filtered RL group leaves a short
                # final accumulation window.
                if micro_since_step < rl_config.gradient_accumulation_steps:
                    correction = rl_config.gradient_accumulation_steps / micro_since_step
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(correction)
                global_update += 1
                schedule_step = base_step + global_update
                multiplier = learning_rate_multiplier(
                    min(global_update - 1, max(0, update_steps - 1)),
                    min(train_config.warmup_steps, max(0, update_steps - 1)),
                    max(1, update_steps),
                    min_ratio=train_config.min_lr_ratio,
                    schedule_type=train_config.schedule_type,
                    decay_fraction=train_config.decay_fraction,
                    decay_shape=train_config.decay_shape,
                )
                optimizer.set_lr_multiplier(multiplier)
                diagnostics = gradient_diagnostics(model) if schedule_step % train_config.diagnostic_interval == 0 else {}
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.max_grad_norm)
                optimizer.step()
                if train_config.qk_clip_interval and schedule_step % train_config.qk_clip_interval == 0:
                    model.apply_qk_clip()
                router_stats = model.update_moe_router_biases()
                optimizer.zero_grad(set_to_none=True)

                loqt_stats = {}
                if train_config.loqt_merge_interval and schedule_step % train_config.loqt_merge_interval == 0:
                    modules = list(iter_loqt_modules(model))
                    parameters = [p for module in modules for p in (module.a, module.b)]
                    merged = merge_loqt_modules(model, on_cpu=train_config.loqt_merge_on_cpu)
                    if train_config.loqt_reset_optimizer_state:
                        clear_optimizer_state(optimizer, parameters)
                    loqt_stats = {"loqt_merged_modules": merged.modules}

                denom = max(1, micro_since_step)
                metrics = {
                    "iteration": args.iteration,
                    "update": global_update,
                    "global_step": schedule_step,
                    "algorithm": rl_config.algorithm,
                    "rollout_records": len(sequences),
                    "tokens_seen": tokens_seen,
                    "loss": running["loss"] / denom,
                    "policy_loss": running["policy"] / denom,
                    "kl_loss": running["kl"] / denom,
                    "router_regularization": running["router"] / denom,
                    "clip_fraction": running["clip"] / denom,
                    "mean_ratio": running["ratio"] / denom,
                    "approx_kl": running["approx_kl"] / denom,
                    "mean_reward": sum(sequence.reward for sequence in sequences) / len(sequences),
                    "grad_norm": float(grad_norm),
                    "lr_multiplier": multiplier,
                    "updates_per_second": global_update / max(time.perf_counter() - started, 1e-9),
                    **router_stats,
                    **diagnostics,
                    **loqt_stats,
                    **sampler.sample(force=True),
                }
                logger.log(metrics)
                print(
                    f"iteration={args.iteration} update={global_update}/{update_steps} "
                    f"loss={metrics['loss']:.5f} reward={metrics['mean_reward']:.3f} "
                    f"clip={metrics['clip_fraction']:.3f}",
                    flush=True,
                )
                micro_since_step = 0
                running = {
                    "loss": 0.0, "policy": 0.0, "kl": 0.0, "router": 0.0,
                    "clip": 0.0, "ratio": 0.0, "approx_kl": 0.0,
                }

        final_step = base_step + global_update
        final = save_checkpoint(
            rl_config.output_dir,
            final_step,
            model,
            optimizer,
            model_config,
            train_config,
            tokens_seen,
            rl_config.keep_last_checkpoints,
        )
        (final / "reasoning_config.json").write_text(json.dumps(rl_config.to_dict(), indent=2), encoding="utf-8")
        print(json.dumps({"checkpoint": str(final), "updates": global_update, "tokens_seen": tokens_seen}, indent=2))
    except BaseException as exc:
        bundle = save_diagnostic_bundle(
            rl_config.output_dir,
            reason="rlvr-failure",
            extra={"iteration": args.iteration, "exception": repr(exc), "tokens_seen": tokens_seen},
        )
        print(f"saved RLVR diagnostics: {bundle}")
        raise


if __name__ == "__main__":
    main()
