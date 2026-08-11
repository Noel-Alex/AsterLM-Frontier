from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .norm import RMSNorm


class GatedKVCompressor(nn.Module):
    """Full-precision semantic reference for DeepSeek V4 gated KV compression.

    The ratio-4 path reproduces V4's overlapping layout: the first half of a
    compressed summary comes from the preceding group and the second half from
    the current group. Other ratios use non-overlapping groups. Incomplete tail
    groups are intentionally not returned because they are not causally visible
    until the group is complete.

    This module is a correctness oracle, not the eventual execution kernel.
    """

    def __init__(
        self,
        input_dim: int,
        head_dim: int,
        compress_ratio: int,
        *,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or head_dim <= 0:
            raise ValueError("input_dim and head_dim must be positive")
        if compress_ratio <= 1:
            raise ValueError("compress_ratio must exceed one")
        self.input_dim = input_dim
        self.head_dim = head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        self.coefficient = 2 if self.overlap else 1

        projected_dim = self.coefficient * head_dim
        self.kv_proj = nn.Linear(input_dim, projected_dim, bias=False, dtype=torch.float32)
        self.gate_proj = nn.Linear(input_dim, projected_dim, bias=False, dtype=torch.float32)
        self.ape = nn.Parameter(
            torch.zeros(compress_ratio, projected_dim, dtype=torch.float32)
        )
        self.norm = RMSNorm(head_dim, eps)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.kv_proj.weight)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.zeros_(self.ape)

    def _overlap_transform(self, tensor: torch.Tensor, fill_value: float) -> torch.Tensor:
        # tensor: [batch, groups, ratio, 2 * head_dim]
        batch, groups, _, _ = tensor.shape
        ratio, dim = self.compress_ratio, self.head_dim
        transformed = tensor.new_full((batch, groups, 2 * ratio, dim), fill_value)
        transformed[:, :, ratio:] = tensor[:, :, :, dim:]
        if groups > 1:
            transformed[:, 1:, :ratio] = tensor[:, :-1, :, :dim]
        return transformed

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3 or hidden.shape[-1] != self.input_dim:
            raise ValueError(
                f"hidden must have shape [batch, sequence, {self.input_dim}]"
            )
        output_dtype = hidden.dtype
        ratio = self.compress_ratio
        complete_tokens = (hidden.shape[1] // ratio) * ratio
        if complete_tokens == 0:
            return hidden.new_empty(hidden.shape[0], 0, self.head_dim)

        # V4 explicitly keeps compression in FP32. Disable any surrounding AMP
        # region so this oracle remains FP32 even when the parent model trains in
        # BF16 autocast.
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            source = hidden[:, :complete_tokens].float()
            kv = F.linear(source, self.kv_proj.weight.float()).unflatten(1, (-1, ratio))
            scores = (
                F.linear(source, self.gate_proj.weight.float()).unflatten(1, (-1, ratio))
                + self.ape.float().view(1, 1, ratio, -1)
            )
            if self.overlap:
                kv = self._overlap_transform(kv, 0.0)
                scores = self._overlap_transform(scores, float("-inf"))
            compressed = (kv * scores.softmax(dim=2)).sum(dim=2)
            compressed = self.norm(compressed)
        return compressed.to(output_dtype)


def compressed_sparse_topk(
    queries: torch.Tensor,
    compressed_keys: torch.Tensor,
    head_weights: torch.Tensor,
    *,
    compress_ratio: int,
    topk: int,
    offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """DeepSeek V4 continuous-score index selection with an exact causal mask.

    ``queries`` must already include the indexer's positional transform and have
    shape ``[B, T, H, D]``. ``compressed_keys`` has shape ``[B, C, D]`` and
    ``head_weights`` has shape ``[B, T, H]``. Returning scores alongside indices
    makes the reference useful for parity tests against future fused kernels.
    Invalid early-query entries are represented by ``-1``.
    """

    if compress_ratio <= 1 or topk <= 0:
        raise ValueError("compress_ratio must exceed one and topk must be positive")
    if queries.ndim != 4 or compressed_keys.ndim != 3 or head_weights.ndim != 3:
        raise ValueError("invalid query, compressed-key, or head-weight rank")
    batch, sequence, heads, head_dim = queries.shape
    if compressed_keys.shape[0] != batch or compressed_keys.shape[2] != head_dim:
        raise ValueError("compressed key shape does not match queries")
    if head_weights.shape != (batch, sequence, heads):
        raise ValueError("head_weights must have shape [B, T, H]")

    compressed_count = compressed_keys.shape[1]
    if compressed_count == 0:
        empty = torch.empty(batch, sequence, 0, dtype=torch.long, device=queries.device)
        return empty, queries.new_empty(batch, sequence, 0)

    scale = head_dim**-0.5 * heads**-0.5
    per_head = torch.einsum(
        "bshd,btd->bsht", queries.float(), compressed_keys.float()
    )
    scores = (per_head.relu() * head_weights.float().unsqueeze(-1) * scale).sum(dim=2)

    positions = torch.arange(sequence, device=queries.device)
    visible_counts = torch.div(positions + 1, compress_ratio, rounding_mode="floor")
    compressed_positions = torch.arange(compressed_count, device=queries.device)
    valid = compressed_positions.view(1, 1, -1) < visible_counts.view(1, -1, 1)
    scores = scores.masked_fill(~valid, float("-inf"))

    selected_count = min(topk, compressed_count)
    selected_scores, indices = scores.topk(selected_count, dim=-1)
    selected_valid = torch.isfinite(selected_scores)
    indices = torch.where(selected_valid, indices + offset, indices.new_full((), -1))
    return indices, selected_scores


def mhc_split_sinkhorn(
    mixes: torch.Tensor,
    scales: torch.Tensor,
    base: torch.Tensor,
    *,
    streams: int = 4,
    iterations: int = 20,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Portable PyTorch equivalent of DeepSeek V4's mHC split/Sinkhorn kernel."""

    expected = (2 + streams) * streams
    if streams <= 0 or iterations <= 0:
        raise ValueError("streams and iterations must be positive")
    if mixes.shape[-1] != expected or base.shape != (expected,) or scales.shape != (3,):
        raise ValueError("mHC parameter geometry is inconsistent")

    pre_logits, post_logits, combination_logits = torch.split(
        mixes, (streams, streams, streams * streams), dim=-1
    )
    pre_base, post_base, combination_base = torch.split(
        base, (streams, streams, streams * streams), dim=-1
    )
    pre = torch.sigmoid(pre_logits * scales[0] + pre_base) + eps
    post = 2.0 * torch.sigmoid(post_logits * scales[1] + post_base)
    combination = (
        combination_logits * scales[2] + combination_base
    ).unflatten(-1, (streams, streams))

    combination = combination.softmax(dim=-1) + eps
    combination = combination / (combination.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iterations - 1):
        combination = combination / (combination.sum(dim=-1, keepdim=True) + eps)
        combination = combination / (combination.sum(dim=-2, keepdim=True) + eps)
    return pre, post, combination


class MHCResidualMixer(nn.Module):
    """Four-stream mHC residual reference for one attention or FFN sublayer."""

    def __init__(
        self,
        dim: int,
        *,
        streams: int = 4,
        iterations: int = 20,
        norm_eps: float = 1e-6,
        sinkhorn_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if dim <= 0 or streams <= 0:
            raise ValueError("dim and streams must be positive")
        self.dim = dim
        self.streams = streams
        self.iterations = iterations
        self.norm_eps = norm_eps
        self.sinkhorn_eps = sinkhorn_eps
        mix_dim = (2 + streams) * streams
        self.mix_weight = nn.Parameter(torch.zeros(mix_dim, streams * dim, dtype=torch.float32))
        self.base = nn.Parameter(torch.empty(mix_dim, dtype=torch.float32))
        self.scales = nn.Parameter(torch.zeros(3, dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.mix_weight)
        with torch.no_grad():
            # Sum of pre weights starts at one; post starts at one. The
            # combination matrix starts close to identity after Sinkhorn.
            pre_probability = 1.0 / self.streams - self.sinkhorn_eps
            pre_logit = math.log(pre_probability / (1.0 - pre_probability))
            self.base[: self.streams].fill_(pre_logit)
            self.base[self.streams : 2 * self.streams].zero_()
            combination = self.base[2 * self.streams :].view(self.streams, self.streams)
            combination.fill_(-6.0)
            combination.diagonal().fill_(6.0)
            self.scales.zero_()

    def reduce(
        self, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if residual.ndim != 4 or residual.shape[-2:] != (self.streams, self.dim):
            raise ValueError(
                f"residual must have shape [B, T, {self.streams}, {self.dim}]"
            )
        output_dtype = residual.dtype
        flat = residual.flatten(2).float()
        inverse_rms = torch.rsqrt(
            flat.square().mean(dim=-1, keepdim=True) + self.norm_eps
        )
        mixes = F.linear(flat, self.mix_weight.float()) * inverse_rms
        pre, post, combination = mhc_split_sinkhorn(
            mixes,
            self.scales.float(),
            self.base.float(),
            streams=self.streams,
            iterations=self.iterations,
            eps=self.sinkhorn_eps,
        )
        reduced = (pre.unsqueeze(-1) * residual.float()).sum(dim=2)
        return reduced.to(output_dtype), post, combination

    def expand(
        self,
        sublayer_output: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        combination: torch.Tensor,
    ) -> torch.Tensor:
        if sublayer_output.shape != residual.shape[:2] + (self.dim,):
            raise ValueError("sublayer output shape does not match residual")
        expanded = post.unsqueeze(-1) * sublayer_output.float().unsqueeze(-2)
        mixed_residual = torch.einsum(
            "bstu,bsud->bstd", combination.float(), residual.float()
        )
        return (expanded + mixed_residual).to(sublayer_output.dtype)


class MHCHeadReducer(nn.Module):
    """DeepSeek V4 head reduction from residual streams to one hidden state."""

    def __init__(
        self,
        dim: int,
        *,
        streams: int = 4,
        norm_eps: float = 1e-6,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if dim <= 0 or streams <= 0:
            raise ValueError("dim and streams must be positive")
        self.dim = dim
        self.streams = streams
        self.norm_eps = norm_eps
        self.eps = eps
        self.mix_weight = nn.Parameter(
            torch.zeros(streams, streams * dim, dtype=torch.float32)
        )
        self.base = nn.Parameter(torch.empty(streams, dtype=torch.float32))
        self.scale = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.mix_weight)
        probability = 1.0 / self.streams - self.eps
        logit = math.log(probability / (1.0 - probability))
        with torch.no_grad():
            self.base.fill_(logit)
            self.scale.zero_()

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        if residual.ndim != 4 or residual.shape[-2:] != (self.streams, self.dim):
            raise ValueError(
                f"residual must have shape [B, T, {self.streams}, {self.dim}]"
            )
        output_dtype = residual.dtype
        flat = residual.flatten(2).float()
        inverse_rms = torch.rsqrt(
            flat.square().mean(dim=-1, keepdim=True) + self.norm_eps
        )
        mixes = F.linear(flat, self.mix_weight.float()) * inverse_rms
        pre = torch.sigmoid(
            mixes * self.scale.float() + self.base.float()
        ) + self.eps
        return (pre.unsqueeze(-1) * residual.float()).sum(dim=2).to(output_dtype)
