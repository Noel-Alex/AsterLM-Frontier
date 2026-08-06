#!/usr/bin/env python
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from tokenizers import Tokenizer, decoders, models, pre_tokenizers

from asterlm.config import AsterConfig
from asterlm.data.tokenizer import SPECIAL_TOKENS
from asterlm.model import AsterLM


def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    root = Path("runs/reasoning-smoke")
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)

    tokens = ["<unk>", *SPECIAL_TOKENS, "What", "is", "2", "+", "?", "4", "think", "answer"]
    while len(tokens) < 64:
        tokens.append(f"tok{len(tokens)}")
    vocab = {token: index for index, token in enumerate(tokens)}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.decoder = decoders.WordPiece(prefix="")
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    tokenizer_path = root / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    model_payload = {
        "model": {
            "vocab_size": 64,
            "d_model": 32,
            "n_layers": 2,
            "n_heads": 4,
            "head_dim": 8,
            "ffn_hidden": 80,
            "ffn_type": "dense",
            "max_seq_len": 128,
            "kda_ratio": 1,
            "kda_backend": "torch",
            "kda_expand_v": 1.0,
            "kda_short_conv": False,
            "latent_rank": 16,
            "q_lora_rank": 16,
            "rope_dim": 8,
            "attention_window": 128,
            "sink_tokens": 2,
            "cache_dtype": "bfloat16",
            "mtp_depth": 0,
            "gradient_checkpointing": True,
        }
    }
    model_config = root / "model.yaml"
    model_config.write_text(yaml.safe_dump(model_payload, sort_keys=False), encoding="utf-8")
    checkpoint = root / "base-checkpoint"
    checkpoint.mkdir()
    model = AsterLM(AsterConfig.from_yaml(model_config))
    torch.save(model.state_dict(), checkpoint / "model.pt")
    (checkpoint / "model_config.yaml").write_text(model_config.read_text(encoding="utf-8"), encoding="utf-8")

    train_config = root / "train.yaml"
    train_config.write_text(
        yaml.safe_dump(
            {
                "train": {
                    "output_dir": str(root / "rl"),
                    "seed": 7,
                    "device": "cpu",
                    "dtype": "float32",
                    "sequence_length": 64,
                    "micro_batch_size": 1,
                    "gradient_accumulation_steps": 1,
                    "max_steps": 2,
                    "optimizer": "muon_adamw",
                    "muon_lr": 0.001,
                    "adam_lr": 0.0001,
                    "warmup_steps": 0,
                    "schedule_type": "constant",
                    "weight_decay": 0.0,
                    "qk_clip_interval": 0,
                    "log_interval": 1,
                    "eval_interval": 100,
                    "eval_batches": 0,
                    "save_interval": 100,
                    "tokenizer_path": str(tokenizer_path),
                    "diagnostic_interval": 1,
                    "system_metrics_interval": 1.0,
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    reasoning_config = root / "reasoning.yaml"
    reasoning_config.write_text(
        yaml.safe_dump(
            {
                "reasoning": {
                    "output_dir": str(root / "rl"),
                    "tokenizer_path": str(tokenizer_path),
                    "seed": 7,
                    "algorithm": "gspo",
                    "iterations": 1,
                    "prompts_per_iteration": 1,
                    "prompt_oversample_factor": 1,
                    "group_size": 2,
                    "update_epochs": 1,
                    "micro_batch_size": 1,
                    "gradient_accumulation_steps": 1,
                    "max_prompt_tokens": 32,
                    "max_completion_tokens": 8,
                    "min_completion_tokens": 1,
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "min_p": 0.0,
                    "dynamic_sampling": False,
                    "kl_beta": 0.0,
                    "force_thinking_prefix": True,
                    "require_reasoning_tags": False,
                    "overlong_buffer_tokens": 2,
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    prompts = root / "prompts.jsonl"
    prompts.write_text(json.dumps({"id": "two-plus-two", "prompt": "What is 2 + 2 ?", "answer": "4", "task": "math"}) + "\n")
    raw = root / "raw.jsonl"
    scored = root / "scored.jsonl"

    run([
        sys.executable, "scripts/generate_reasoning_rollouts.py",
        "--model", str(model_config), "--checkpoint", str(checkpoint),
        "--reasoning", str(reasoning_config), "--prompts", str(prompts),
        "--output", str(raw), "--device", "cpu",
    ])
    run([
        sys.executable, "scripts/score_reasoning_rollouts.py",
        "--input", str(raw), "--output", str(scored), "--reasoning", str(reasoning_config),
    ])
    run([
        sys.executable, "scripts/train_rlvr.py",
        "--model", str(model_config), "--train", str(train_config),
        "--reasoning", str(reasoning_config), "--checkpoint", str(checkpoint),
        "--rollouts", str(scored), "--iteration", "0",
    ])
    latest = root / "rl" / "latest.txt"
    if not latest.exists():
        raise RuntimeError("RLVR smoke test produced no checkpoint")
    print(json.dumps({"status": "ok", "checkpoint": latest.read_text().strip()}, indent=2))


if __name__ == "__main__":
    main()
