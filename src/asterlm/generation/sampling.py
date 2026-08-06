from __future__ import annotations

import torch


def apply_repetition_penalty(logits: torch.Tensor, token_ids: torch.Tensor, penalty: float) -> torch.Tensor:
    if penalty == 1.0 or token_ids.numel() == 0:
        return logits
    unique = torch.unique(token_ids, sorted=False)
    selected = logits.index_select(-1, unique)
    selected = torch.where(selected < 0, selected * penalty, selected / penalty)
    logits = logits.clone()
    logits.scatter_(-1, unique, selected)
    return logits


def filter_logits(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    min_p: float = 0.0,
) -> torch.Tensor:
    filtered = logits
    if top_k > 0 and top_k < filtered.shape[-1]:
        threshold = torch.topk(filtered, top_k, dim=-1).values[..., -1, None]
        filtered = filtered.masked_fill(filtered < threshold, float("-inf"))
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        cumulative = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        filtered = torch.full_like(filtered, float("-inf")).scatter(-1, sorted_indices, sorted_logits)
    if min_p > 0.0:
        probabilities = filtered.softmax(dim=-1)
        cutoff = probabilities.amax(dim=-1, keepdim=True) * min_p
        filtered = filtered.masked_fill(probabilities < cutoff, float("-inf"))
    return filtered


def sample_next(
    logits: torch.Tensor,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.95,
    min_p: float = 0.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1)
    logits = logits / temperature
    logits = filter_logits(logits, top_k=top_k, top_p=top_p, min_p=min_p)
    return torch.multinomial(logits.softmax(dim=-1), num_samples=1, generator=generator).squeeze(-1)
