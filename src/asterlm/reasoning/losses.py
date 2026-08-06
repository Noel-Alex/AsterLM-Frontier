from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.checkpoint import checkpoint


@dataclass(slots=True)
class RLVRLossOutput:
    loss: torch.Tensor
    policy_loss: torch.Tensor
    kl_loss: torch.Tensor
    entropy_proxy: torch.Tensor
    clip_fraction: torch.Tensor
    mean_ratio: torch.Tensor
    approx_kl: torch.Tensor


def compute_group_advantages(
    rewards: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    algorithm: str,
    eps: float = 1e-6,
    normalize_std: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-sample advantages and per-group reward standard deviations."""

    if rewards.ndim != 1 or group_ids.shape != rewards.shape:
        raise ValueError("rewards and group_ids must be one-dimensional and aligned")
    advantages = torch.zeros_like(rewards, dtype=torch.float32)
    unique = torch.unique(group_ids, sorted=True)
    stds = []
    for group in unique:
        mask = group_ids.eq(group)
        group_rewards = rewards[mask].float()
        centered = group_rewards - group_rewards.mean()
        std = group_rewards.std(unbiased=False)
        stds.append(std)
        # Dr.GRPO removes the sample-standard-deviation division that can amplify
        # nearly constant groups.  DAPO/GRPO/GSPO may use standardized advantages.
        if algorithm == "dr_grpo" or not normalize_std:
            advantages[mask] = centered
        else:
            advantages[mask] = centered / (std + eps)
    return advantages, torch.stack(stds) if stds else torch.empty(0, device=rewards.device)


def selected_token_logprobs(
    logits: torch.Tensor,
    targets: torch.Tensor,
    completion_mask: torch.Tensor,
) -> torch.Tensor:
    if logits.shape[:-1] != targets.shape or targets.shape != completion_mask.shape:
        raise ValueError("logits, targets, and mask shapes are inconsistent")
    safe_targets = targets.masked_fill(~completion_mask, 0)
    logps = logits.float().log_softmax(dim=-1)
    return logps.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1) * completion_mask


def selected_token_logprobs_from_hidden(
    model: torch.nn.Module,
    hidden_states: torch.Tensor,
    targets: torch.Tensor,
    completion_mask: torch.Tensor,
    *,
    chunk_size: int,
    checkpoint_chunks: bool = True,
) -> torch.Tensor:
    """Compute only target-token log-probabilities without retaining full logits.

    Reasoning RL needs log p(target) rather than the complete vocabulary distribution.
    Projecting hidden states in short sequence chunks bounds the largest temporary
    allocation at ``[batch, chunk_size, vocabulary]`` instead of
    ``[batch, full_sequence, vocabulary]``.  Checkpointing recomputes those short
    projections during backward and avoids retaining their softmax intermediates.
    """

    if hidden_states.shape[:-1] != targets.shape or targets.shape != completion_mask.shape:
        raise ValueError("hidden states, targets, and mask shapes are inconsistent")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    safe_targets = targets.masked_fill(~completion_mask, 0)

    def chunk_logps(h: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        projected = model.embedding_out_proj(h)
        logits = model.lm_head(projected).float()
        selected = logits.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
        return selected - torch.logsumexp(logits, dim=-1)

    pieces: list[torch.Tensor] = []
    for start in range(0, hidden_states.shape[1], chunk_size):
        stop = min(start + chunk_size, hidden_states.shape[1])
        hidden_chunk = hidden_states[:, start:stop]
        target_chunk = safe_targets[:, start:stop]
        if checkpoint_chunks and model.training and torch.is_grad_enabled():
            values = checkpoint(chunk_logps, hidden_chunk, target_chunk, use_reentrant=False)
        else:
            values = chunk_logps(hidden_chunk, target_chunk)
        pieces.append(values)
    return torch.cat(pieces, dim=1) * completion_mask


def _masked_sample_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1)


def rlvr_policy_loss(
    current_token_logps: torch.Tensor,
    old_token_logps: torch.Tensor,
    completion_mask: torch.Tensor,
    advantages: torch.Tensor,
    *,
    algorithm: str,
    clip_low: float,
    clip_high: float,
    ratio_log_clip: float = 20.0,
    reference_token_logps: torch.Tensor | None = None,
    kl_beta: float = 0.0,
    entropy_bonus: float = 0.0,
) -> RLVRLossOutput:
    """Compute GSPO, DAPO, GRPO, Dr.GRPO, or REINFORCE-baseline loss.

    All old-policy and optional reference log-probabilities are precomputed.  This
    is what permits exact on-policy updates without a second full model in VRAM.
    """

    if current_token_logps.shape != old_token_logps.shape or current_token_logps.shape != completion_mask.shape:
        raise ValueError("token log-probability tensors and mask must align")
    if advantages.ndim != 1 or advantages.shape[0] != current_token_logps.shape[0]:
        raise ValueError("advantages must have one value per sequence")
    mask = completion_mask.to(current_token_logps.dtype)
    log_ratio = (current_token_logps - old_token_logps) * mask
    token_ratio = torch.exp(log_ratio.clamp(-ratio_log_clip, ratio_log_clip))
    low, high = 1.0 - clip_low, 1.0 + clip_high
    clipped_token_ratio = token_ratio.clamp(low, high)
    advantage = advantages.to(current_token_logps.dtype)

    if algorithm == "gspo":
        sequence_log_ratio = _masked_sample_mean(log_ratio, mask)
        ratio = torch.exp(sequence_log_ratio.clamp(-ratio_log_clip, ratio_log_clip))
        clipped = ratio.clamp(low, high)
        surrogate = torch.minimum(ratio * advantage, clipped * advantage)
        policy_loss = -surrogate.mean()
        clip_fraction = ratio.ne(clipped).float().mean()
        mean_ratio = ratio.mean()
    elif algorithm == "reinforce_baseline":
        sequence_logp = _masked_sample_mean(current_token_logps, mask)
        policy_loss = -(advantage * sequence_logp).mean()
        clip_fraction = torch.zeros((), device=current_token_logps.device)
        mean_ratio = torch.exp(_masked_sample_mean(log_ratio, mask)).mean()
    else:
        unclipped = token_ratio * advantage[:, None]
        clipped = clipped_token_ratio * advantage[:, None]
        token_objective = torch.minimum(unclipped, clipped) * mask
        if algorithm in {"dapo", "dr_grpo"}:
            # DAPO's token-level policy-gradient loss avoids giving every response
            # equal weight regardless of length. Dr.GRPO uses the same constant/global
            # normalization rather than a per-response length denominator.
            policy_loss = -token_objective.sum() / mask.sum().clamp_min(1)
        elif algorithm == "grpo":
            policy_loss = -_masked_sample_mean(token_objective, mask).mean()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        clip_fraction = ((token_ratio.ne(clipped_token_ratio)).to(mask.dtype) * mask).sum() / mask.sum().clamp_min(1)
        mean_ratio = (token_ratio * mask).sum() / mask.sum().clamp_min(1)

    if reference_token_logps is not None and kl_beta > 0:
        if reference_token_logps.shape != current_token_logps.shape:
            raise ValueError("reference token log-probabilities must align")
        log_ref_ratio = (reference_token_logps - current_token_logps) * mask
        # Schulman's non-negative k3 estimator: exp(x) - x - 1 where x=log(pi_ref/pi).
        token_kl = torch.exp(log_ref_ratio.clamp(-ratio_log_clip, ratio_log_clip)) - log_ref_ratio - 1.0
        kl_loss = (token_kl * mask).sum() / mask.sum().clamp_min(1)
    else:
        kl_loss = torch.zeros((), device=current_token_logps.device)

    # Selected-token surprise is only an entropy proxy; full-vocabulary entropy would
    # require retaining another [B,T,V] tensor. It is logged and disabled by default.
    entropy_proxy = -(current_token_logps * mask).sum() / mask.sum().clamp_min(1)
    total = policy_loss + kl_beta * kl_loss - entropy_bonus * entropy_proxy
    approx_kl = ((old_token_logps - current_token_logps) * mask).sum() / mask.sum().clamp_min(1)
    return RLVRLossOutput(total, policy_loss, kl_loss, entropy_proxy, clip_fraction, mean_ratio, approx_kl)
