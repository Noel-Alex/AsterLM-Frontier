from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class RLVRConfig:
    """Single-GPU, verifier-rewarded reasoning RL configuration.

    Rollout and update are intentionally decoupled.  Only one full model needs to
    reside on the GPU at any time; an optional reference model is scored in a
    separate process and its token log-probabilities are stored with the rollouts.
    """

    output_dir: str = "runs/reasoning-rlvr"
    tokenizer_path: str = "artifacts/tokenizer.json"
    seed: int = 1337

    algorithm: str = "gspo"  # gspo | dapo | grpo | dr_grpo | reinforce_baseline
    iterations: int = 200
    prompts_per_iteration: int = 8
    prompt_oversample_factor: int = 2
    group_size: int = 4
    update_epochs: int = 1
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 8

    max_prompt_tokens: int = 2048
    max_completion_tokens: int = 4096
    min_completion_tokens: int = 32
    temperature: float = 0.8
    top_k: int = 0
    top_p: float = 0.95
    min_p: float = 0.02
    repetition_penalty: float = 1.0
    prefill_chunk_size: int = 2048

    clip_low: float = 0.20
    clip_high: float = 0.28
    kl_beta: float = 0.0
    entropy_bonus: float = 0.0
    advantage_epsilon: float = 1e-6
    ratio_log_clip: float = 20.0
    normalize_group_std: bool = True
    dynamic_sampling: bool = True
    min_group_reward_std: float = 1e-6
    max_groups_per_update: int = 0  # 0 = all usable groups

    correctness_weight: float = 1.0
    format_weight: float = 0.10
    reasoning_complete_weight: float = 0.05
    invalid_penalty: float = 0.25
    repetition_penalty_reward: float = 0.10
    overlong_buffer_tokens: int = 256
    overlong_penalty: float = 0.25
    reward_clip_min: float = -1.0
    reward_clip_max: float = 1.25

    force_thinking_prefix: bool = True
    require_reasoning_tags: bool = True
    direct_mode_fraction: float = 0.0
    answer_style: str = "boxed"  # boxed | answer_line | xml

    checkpoint_every_iterations: int = 1
    keep_last_checkpoints: int = 3
    save_rollouts: bool = True
    allow_unsafe_code_verifier: bool = False
    code_timeout_seconds: float = 4.0
    code_memory_mb: int = 512

    def __post_init__(self) -> None:
        if self.algorithm not in {"gspo", "dapo", "grpo", "dr_grpo", "reinforce_baseline"}:
            raise ValueError("Unsupported RLVR algorithm")
        if self.answer_style not in {"boxed", "answer_line", "xml"}:
            raise ValueError("answer_style must be boxed, answer_line, or xml")
        for name in (
            "iterations",
            "prompts_per_iteration",
            "group_size",
            "update_epochs",
            "micro_batch_size",
            "gradient_accumulation_steps",
            "max_prompt_tokens",
            "max_completion_tokens",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.prompt_oversample_factor <= 0:
            raise ValueError("prompt_oversample_factor must be positive")
        if self.min_completion_tokens < 0 or self.min_completion_tokens > self.max_completion_tokens:
            raise ValueError("Invalid completion token limits")
        if not 0 < self.temperature <= 5:
            raise ValueError("temperature must be in (0, 5]")
        if not 0 <= self.top_p <= 1 or not 0 <= self.min_p <= 1:
            raise ValueError("top_p/min_p must be in [0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")
        if not 0 <= self.direct_mode_fraction <= 1:
            raise ValueError("direct_mode_fraction must be in [0, 1]")
        if self.clip_low < 0 or self.clip_high < 0:
            raise ValueError("clipping bounds must be non-negative")
        if self.reward_clip_min >= self.reward_clip_max:
            raise ValueError("reward clip bounds are invalid")
        if self.code_timeout_seconds <= 0 or self.code_memory_mb <= 0:
            raise ValueError("code verifier limits must be positive")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RLVRConfig":
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if "reasoning" in payload:
            payload = payload["reasoning"]
        allowed = {field.name for field in fields(cls)}
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"Unknown RLVRConfig keys: {sorted(unknown)}")
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
