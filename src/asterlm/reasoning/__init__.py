"""Reasoning-model post-training utilities for AsterLM."""

from .config import RLVRConfig
from .formatting import (
    ReasoningMode,
    extract_final_answer,
    format_reasoning_prompt,
    split_reasoning_answer,
)
from .losses import (
    RLVRLossOutput,
    compute_group_advantages,
    rlvr_policy_loss,
    selected_token_logprobs_from_hidden,
)
from .verifiers import RewardBreakdown, score_completion

__all__ = [
    "RLVRConfig",
    "ReasoningMode",
    "extract_final_answer",
    "format_reasoning_prompt",
    "split_reasoning_answer",
    "RLVRLossOutput",
    "compute_group_advantages",
    "rlvr_policy_loss",
    "selected_token_logprobs_from_hidden",
    "RewardBreakdown",
    "score_completion",
]
