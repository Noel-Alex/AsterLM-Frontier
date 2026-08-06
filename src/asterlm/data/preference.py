from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .tokenizer import AsterTokenizer, format_chat, normalize_messages


@dataclass
class PreferenceSequence:
    input_ids: torch.Tensor
    target_ids: torch.Tensor
    response_mask: torch.Tensor


def _prompt_text(prompt: Any) -> str:
    if isinstance(prompt, list):
        return format_chat(prompt, add_generation_prompt=True)
    return format_chat([{"role": "user", "content": str(prompt)}], add_generation_prompt=True)


def _response_text(response: Any) -> str:
    """Normalize preference responses, including UltraFeedback message lists."""
    if isinstance(response, str):
        return response
    if isinstance(response, list):
        messages = normalize_messages(response)
        for message in reversed(messages):
            if message.get("role") == "assistant":
                return str(message.get("content", ""))
        if messages:
            return str(messages[-1].get("content", ""))
        return ""
    if isinstance(response, dict):
        for key in ("content", "text", "value", "response"):
            if response.get(key) is not None:
                return str(response[key])
    return str(response)


def encode_preference_sequence(
    tokenizer: AsterTokenizer,
    prompt: Any,
    response: Any,
    max_length: int,
) -> PreferenceSequence:
    prompt_ids = tokenizer.encode(_prompt_text(prompt))
    response_ids = tokenizer.encode(_response_text(response)) + [tokenizer.token_to_id("<|end|>")]
    ids = prompt_ids + response_ids
    flags = [False] * len(prompt_ids) + [True] * len(response_ids)
    if len(ids) > max_length + 1:
        # Preserve the complete response where possible; trim oldest prompt tokens first.
        overflow = len(ids) - (max_length + 1)
        trim_prompt = min(overflow, max(0, len(prompt_ids) - 8))
        ids = ids[trim_prompt:]
        flags = flags[trim_prompt:]
        if len(ids) > max_length + 1:
            ids = ids[: max_length + 1]
            flags = flags[: max_length + 1]
    return PreferenceSequence(
        input_ids=torch.tensor(ids[:-1], dtype=torch.long),
        target_ids=torch.tensor(ids[1:], dtype=torch.long),
        response_mask=torch.tensor(flags[1:], dtype=torch.bool),
    )


def pad_preference_batch(
    sequences: list[PreferenceSequence], pad_id: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    length = max(sequence.input_ids.numel() for sequence in sequences)
    batch = len(sequences)
    input_ids = torch.full((batch, length), pad_id, dtype=torch.long)
    targets = torch.full((batch, length), pad_id, dtype=torch.long)
    masks = torch.zeros((batch, length), dtype=torch.bool)
    for index, sequence in enumerate(sequences):
        n = sequence.input_ids.numel()
        input_ids[index, :n] = sequence.input_ids
        targets[index, :n] = sequence.target_ids
        masks[index, :n] = sequence.response_mask
    return input_ids, targets, masks


def response_logprobs(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    log_probs = logits.float().log_softmax(dim=-1)
    token_logps = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    masked = token_logps * mask
    return masked.sum(dim=-1), mask.sum(dim=-1).clamp_min(1)
