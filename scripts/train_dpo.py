#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from asterlm import AsterConfig, AsterLM, TrainConfig
from asterlm.data import AsterTokenizer
from asterlm.data.preference import encode_preference_sequence, pad_preference_batch, response_logprobs
from asterlm.optim import build_optimizer, learning_rate_multiplier
from asterlm.quantization.loqt import iter_loqt_modules, merge_loqt_modules
from asterlm.training.checkpoint import (
    load_checkpoint,
    load_model_weights,
    pin_kda_backend_from_checkpoint,
    save_checkpoint,
)
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
    parser = argparse.ArgumentParser(description="Offline-reference, single-model DPO for AsterLM")
    parser.add_argument("--model", default="configs/model/aster_moe_frontier_893m_a484m.yaml")
    parser.add_argument("--train", default="configs/train/dpo_laptop.yaml")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", help="SFT checkpoint for a new DPO run")
    group.add_argument("--resume", help="Resume a DPO checkpoint with optimizer and RNG state")
    parser.add_argument("--data", required=True, help="Reference-scored JSONL")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--sft-weight", type=float, default=0.05)
    parser.add_argument("--max-length", type=int, default=4096)
    args = parser.parse_args()

    checkpoint_source = args.resume or args.checkpoint
    assert checkpoint_source is not None
    model_config = AsterConfig.from_yaml(args.model)
    pin_kda_backend_from_checkpoint(model_config, checkpoint_source)
    train_config = TrainConfig.from_yaml(args.train)
    if args.resume:
        train_config.resume = args.resume
    random.seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_config.seed)
    device = torch.device(train_config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if args.max_length > model_config.max_seq_len:
        raise ValueError("--max-length exceeds model max_seq_len")
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[train_config.dtype]
    if device.type == "cuda" and dtype == torch.float16:
        raise ValueError("Use BF16 rather than unscaled FP16 on RTX 4080")
    if (
        train_config.precision_backend == "transformer_engine_fp8"
        and model_config.linear_backend != "transformer_engine"
    ):
        raise ValueError("FP8 DPO requires model.linear_backend=transformer_engine")

    tokenizer = AsterTokenizer(train_config.tokenizer_path)
    pad_id = tokenizer.token_to_id("<|pad|>")
    model = AsterLM(model_config)
    model = model_to_training_dtype(model, device, dtype)
    precision = PrecisionManager(train_config, device, dtype)
    optimizer = build_optimizer(model, train_config)
    if args.resume:
        step, tokens_seen = load_checkpoint(model, optimizer, args.resume)
    else:
        load_model_weights(model, checkpoint_source)
        step, tokens_seen = 0, 0

    records = [
        json.loads(line)
        for line in Path(args.data).open("r", encoding="utf-8")
        if line.strip()
    ]
    required = {"prompt", "chosen", "rejected", "ref_chosen_logp", "ref_rejected_logp"}
    if not records or not required.issubset(records[0]):
        raise ValueError(f"DPO data must contain {sorted(required)}")
    output = Path(train_config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(output / "dpo_metrics.jsonl")
    sampler = SystemSampler(device, min_interval=train_config.system_metrics_interval)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    tokens_per_update = args.max_length * 2 * train_config.gradient_accumulation_steps
    effective_steps = min(
        train_config.max_steps,
        math.ceil(train_config.max_tokens / tokens_per_update)
        if train_config.max_tokens is not None
        else train_config.max_steps,
    )

    try:
        for update in range(step, effective_steps):
            if train_config.max_tokens is not None and tokens_seen >= train_config.max_tokens:
                break
            multiplier = learning_rate_multiplier(
                update,
                train_config.warmup_steps,
                effective_steps,
                min_ratio=train_config.min_lr_ratio,
                schedule_type=train_config.schedule_type,
                decay_fraction=train_config.decay_fraction,
                decay_shape=train_config.decay_shape,
            )
            optimizer.set_lr_multiplier(multiplier)
            total = 0.0
            reward_margin = 0.0
            for _ in range(train_config.gradient_accumulation_steps):
                record = random.choice(records)
                chosen = encode_preference_sequence(
                    tokenizer, record["prompt"], record["chosen"], args.max_length
                )
                rejected = encode_preference_sequence(
                    tokenizer, record["prompt"], record["rejected"], args.max_length
                )
                input_ids, targets, masks = pad_preference_batch([chosen, rejected], pad_id)
                input_ids = input_ids.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                with precision.activation_context(), precision.forward_context():
                    logits = model(input_ids).logits
                    if logits is None:
                        raise RuntimeError("DPO requires logits")
                    logps, lengths = response_logprobs(logits, targets, masks)
                    policy_ratio = logps[0] - logps[1]
                    reference_ratio = torch.as_tensor(
                        float(record["ref_chosen_logp"]) - float(record["ref_rejected_logp"]),
                        device=device,
                    )
                    dpo_loss = -F.logsigmoid(args.beta * (policy_ratio - reference_ratio))
                    chosen_nll = -logps[0] / lengths[0]
                    loss = dpo_loss + args.sft_weight * chosen_nll
                (loss / train_config.gradient_accumulation_steps).backward()
                total += float(loss.detach())
                reward_margin += float((args.beta * (policy_ratio - reference_ratio)).detach())
                tokens_seen += int(masks.sum())

            diagnostics = (
                gradient_diagnostics(model)
                if (update + 1) % train_config.diagnostic_interval == 0
                else {}
            )
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.max_grad_norm)
            optimizer.step()
            if train_config.qk_clip_interval and (update + 1) % train_config.qk_clip_interval == 0:
                model.apply_qk_clip()
            optimizer.zero_grad(set_to_none=True)
            step = update + 1

            loqt_stats = {}
            if train_config.loqt_merge_interval and step % train_config.loqt_merge_interval == 0:
                modules = list(iter_loqt_modules(model))
                parameters = [p for module in modules for p in (module.a, module.b)]
                merge = merge_loqt_modules(model, on_cpu=train_config.loqt_merge_on_cpu)
                if train_config.loqt_reset_optimizer_state:
                    clear_optimizer_state(optimizer, parameters)
                loqt_stats = {"loqt_merged_modules": merge.modules}

            if step % train_config.log_interval == 0:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
                metrics = {
                    "step": step,
                    "tokens_seen": tokens_seen,
                    "dpo_total_loss": total / train_config.gradient_accumulation_steps,
                    "reward_margin": reward_margin / train_config.gradient_accumulation_steps,
                    "grad_norm": float(grad_norm),
                    "lr_multiplier": multiplier,
                    "updates_per_second": step / max(elapsed, 1e-9),
                    **diagnostics,
                    **loqt_stats,
                    **sampler.sample(force=True),
                }
                logger.log(metrics)
                print(
                    f"step={step} loss={metrics['dpo_total_loss']:.4f} "
                    f"margin={metrics['reward_margin']:.4f}"
                )
            if step % train_config.save_interval == 0:
                save_checkpoint(
                    train_config.output_dir,
                    step,
                    model,
                    optimizer,
                    model_config,
                    train_config,
                    tokens_seen,
                    train_config.keep_last_checkpoints,
                )
        final = save_checkpoint(
            train_config.output_dir,
            step,
            model,
            optimizer,
            model_config,
            train_config,
            tokens_seen,
            train_config.keep_last_checkpoints,
        )
        print(f"DPO complete: {final}")
    except BaseException as exc:
        bundle = save_diagnostic_bundle(
            train_config.output_dir,
            reason="dpo-failure",
            extra={"step": step, "tokens_seen": tokens_seen, "exception": repr(exc)},
        )
        print(f"saved DPO failure diagnostics: {bundle}")
        raise


if __name__ == "__main__":
    main()
