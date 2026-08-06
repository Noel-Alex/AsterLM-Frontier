from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from asterlm.generation.sampling import apply_repetition_penalty, sample_next
from asterlm.model import AsterLM


@dataclass(slots=True)
class RolloutResult:
    completion_ids: list[int]
    old_token_logps: list[float]
    finish_reason: str


@torch.inference_mode()
def sample_rollout(
    model: AsterLM,
    prompt_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    min_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    min_p: float,
    repetition_penalty: float,
    eos_token_ids: Iterable[int],
    seed: int,
    prefill_chunk_size: int = 2048,
) -> RolloutResult:
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
        raise ValueError("prompt_ids must have shape [1, sequence]")
    if prompt_ids.shape[1] >= model.config.max_seq_len:
        raise ValueError("Prompt already reaches the model context limit")
    max_new_tokens = min(max_new_tokens, model.config.max_seq_len - prompt_ids.shape[1])
    if max_new_tokens <= 0:
        return RolloutResult([], [], "context_limit")

    device = prompt_ids.device
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    eos = set(int(token) for token in eos_token_ids)
    cache = model.make_cache()
    output = None
    chunk_size = max(1, prefill_chunk_size)
    for start in range(0, prompt_ids.shape[1], chunk_size):
        output = model(prompt_ids[:, start : start + chunk_size], cache=cache, use_cache=True)
    assert output is not None and output.logits is not None
    logits = output.logits[:, -1]
    history = prompt_ids[0]
    completion: list[int] = []
    selected_logps: list[float] = []
    finish_reason = "length"

    for step in range(max_new_tokens):
        penalized = apply_repetition_penalty(logits[0], history, repetition_penalty)
        token = sample_next(
            penalized.unsqueeze(0),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            generator=generator,
        )
        value = int(token.item())
        logp = torch.log_softmax(logits.float(), dim=-1)[0, value]
        completion.append(value)
        selected_logps.append(float(logp))
        history = torch.cat((history, token.view(1)))
        if step + 1 >= min_new_tokens and value in eos:
            finish_reason = "eos"
            break
        output = model(token.view(1, 1), cache=cache, use_cache=True)
        if output.logits is None:
            raise RuntimeError("Rollout model did not return logits")
        logits = output.logits[:, -1]
    return RolloutResult(completion, selected_logps, finish_reason)
