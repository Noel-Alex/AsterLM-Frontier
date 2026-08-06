from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from asterlm.cache import LatentLayerCache
from asterlm.config import AsterConfig
from .linear import build_linear, mark_residual
from .norm import build_norm
from .rotary import RotaryEmbedding, apply_rotary


class LatentAttention(nn.Module):
    """DeepSeek-MLA-inspired global attention with a compressed latent cache.

    K/V content is represented by one low-rank latent vector per token plus a separate
    RoPE key channel. One-token decoding absorbs K/V up-projections and performs online
    softmax over hot/quantized cache chunks, avoiding a reconstructed full KV cache.
    """

    def __init__(self, config: AsterConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.rope_dim = config.rope_dim
        self.latent_rank = config.latent_rank
        self.q_lora_rank = config.q_lora_rank
        self.window = config.attention_window
        self.sink_tokens = config.sink_tokens
        self.dropout_p = config.attention_dropout
        self.logit_softcap = config.logit_softcap
        self.qk_stat_tokens = config.qk_stat_tokens

        q_dim = self.n_heads * (self.head_dim + self.rope_dim)
        backend = config.linear_backend
        if self.q_lora_rank is None:
            self.q_down = None
            self.q_norm = None
            self.q_up = build_linear(self.d_model, q_dim, bias=False, backend=backend)
        else:
            self.q_down = build_linear(self.d_model, self.q_lora_rank, bias=False, backend=backend)
            self.q_norm = build_norm(self.q_lora_rank, config.rms_eps, config.norm_type)
            self.q_up = build_linear(self.q_lora_rank, q_dim, bias=False, backend=backend)
        self.kv_down = build_linear(
            self.d_model, self.latent_rank + self.rope_dim, bias=False, backend=backend
        )
        self.latent_norm = (
            build_norm(self.latent_rank, config.rms_eps, config.norm_type)
            if config.latent_rms_norm
            else nn.Identity()
        )
        self.k_up = build_linear(self.latent_rank, self.d_model, bias=False, backend=backend)
        self.v_up = build_linear(self.latent_rank, self.d_model, bias=False, backend=backend)
        self.out_proj = mark_residual(
            build_linear(self.d_model, self.d_model, bias=False, backend=backend)
        )
        self.gate_proj = (
            build_linear(self.d_model, self.d_model, bias=True, backend=backend)
            if config.attention_gate
            else None
        )

        self.rope = RotaryEmbedding(
            dim=self.rope_dim,
            theta=config.rope_theta,
            max_position=config.max_seq_len,
            scaling_type=config.rope_scaling_type,
            scaling_factor=config.rope_scaling_factor,
            original_max_position=config.rope_original_max_position,
            beta_fast=config.yarn_beta_fast,
            beta_slow=config.yarn_beta_slow,
        )
        self.register_buffer("last_max_logits", torch.zeros(self.n_heads), persistent=False)

    def _query(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.q_down is None:
            return self.q_up(hidden)
        assert self.q_norm is not None
        return self.q_up(self.q_norm(self.q_down(hidden)))

    def _project(
        self, hidden: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = hidden.shape
        q = self._query(hidden).view(bsz, seq_len, self.n_heads, self.head_dim + self.rope_dim)
        q_content, q_rope = q.split((self.head_dim, self.rope_dim), dim=-1)
        compressed = self.kv_down(hidden)
        latent_raw, k_rope = compressed.split((self.latent_rank, self.rope_dim), dim=-1)
        latent = self.latent_norm(latent_raw)
        cos, sin = self.rope(q_rope, position_ids)
        q_rope = apply_rotary(q_rope, cos, sin)
        k_rope = apply_rotary(k_rope.unsqueeze(2), cos, sin).squeeze(2)
        return q_content, q_rope, latent, k_rope, compressed

    @torch.no_grad()
    def _record_qk_statistics(
        self, q_content: torch.Tensor, q_rope: torch.Tensor, k_content: torch.Tensor, k_rope: torch.Tensor
    ) -> None:
        if not self.training or self.qk_stat_tokens <= 0:
            return
        t = q_content.shape[1]
        n = min(t, self.qk_stat_tokens)
        idx = torch.linspace(0, t - 1, n, device=q_content.device).round().long()
        q = torch.cat((q_content[:, idx], q_rope[:, idx]), dim=-1)
        kr = k_rope[:, idx].unsqueeze(2).expand(-1, -1, self.n_heads, -1)
        k = torch.cat((k_content[:, idx], kr), dim=-1)
        logits = torch.einsum("bihd,bjhd->bhij", q.float(), k.float()) / math.sqrt(q.shape[-1])
        maxima = logits.amax(dim=(0, 2, 3)).detach().to(self.last_max_logits.device)
        self.last_max_logits.copy_(torch.maximum(self.last_max_logits, maxima))

    def _full_attention(
        self,
        q_content: torch.Tensor,
        q_rope: torch.Tensor,
        latent: torch.Tensor,
        k_rope: torch.Tensor,
        previous: LatentLayerCache | None,
    ) -> torch.Tensor:
        bsz, q_len, _, _ = q_content.shape
        if previous is not None and previous.length:
            previous_latent = previous.materialize_latent(dtype=latent.dtype, device=latent.device)
            previous_rope = previous.materialize_rope(dtype=k_rope.dtype, device=k_rope.device)
            assert previous_latent is not None and previous_rope is not None
            all_latent = torch.cat((previous_latent, latent), dim=1)
            all_k_rope = torch.cat((previous_rope, k_rope), dim=1)
            past_len = previous.length
        else:
            all_latent = latent
            all_k_rope = k_rope
            past_len = 0

        k_content_all = self.k_up(all_latent).view(bsz, -1, self.n_heads, self.head_dim)
        value_all = self.v_up(all_latent).view(bsz, -1, self.n_heads, self.head_dim)
        k_rope_all = all_k_rope.unsqueeze(2).expand(-1, -1, self.n_heads, -1)
        query = torch.cat((q_content, q_rope), dim=-1).transpose(1, 2)
        key = torch.cat((k_content_all, k_rope_all), dim=-1).transpose(1, 2)
        value = value_all.transpose(1, 2)

        current_k = self.k_up(latent).view(bsz, q_len, self.n_heads, self.head_dim)
        self._record_qk_statistics(q_content, q_rope, current_k, k_rope)

        total = key.shape[-2]
        if past_len == 0:
            allowed = None
        else:
            q_index = torch.arange(q_len, device=query.device).unsqueeze(1)
            k_index = torch.arange(total, device=query.device).unsqueeze(0)
            allowed = k_index <= (past_len + q_index)

        if self.logit_softcap is None:
            out = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=allowed,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=past_len == 0 and q_len > 1,
            )
        else:
            scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(query.shape[-1])
            scores = self.logit_softcap * torch.tanh(scores / self.logit_softcap)
            if past_len == 0 and q_len > 1:
                causal = torch.ones(q_len, total, dtype=torch.bool, device=query.device).tril()
                scores = scores.masked_fill(~causal, float("-inf"))
            elif allowed is not None:
                scores = scores.masked_fill(~allowed, float("-inf"))
            weights = scores.float().softmax(dim=-1).to(query.dtype)
            weights = F.dropout(weights, p=self.dropout_p, training=self.training)
            out = torch.matmul(weights, value)
        return out.transpose(1, 2)

    def _absorbed_decode(
        self,
        q_content: torch.Tensor,
        q_rope: torch.Tensor,
        latent: torch.Tensor,
        k_rope: torch.Tensor,
        previous: LatentLayerCache,
    ) -> torch.Tensor:
        """Exact online softmax over quantized chunks with O(chunk) temporary memory."""
        wk = self.k_up.weight.view(self.n_heads, self.head_dim, self.latent_rank)
        wv = self.v_up.weight.view(self.n_heads, self.head_dim, self.latent_rank)
        q_latent = torch.einsum("bthd,hdl->bthl", q_content, wk)
        scale = 1.0 / math.sqrt(self.head_dim + self.rope_dim)
        running_max: torch.Tensor | None = None
        running_sum: torch.Tensor | None = None
        running_context: torch.Tensor | None = None

        def consume(chunk_latent: torch.Tensor, chunk_rope: torch.Tensor) -> None:
            nonlocal running_max, running_sum, running_context
            scores = (
                torch.einsum("bthl,bsl->bhts", q_latent, chunk_latent)
                + torch.einsum("bthr,bsr->bhts", q_rope, chunk_rope)
            ) * scale
            if self.logit_softcap is not None:
                scores = self.logit_softcap * torch.tanh(scores / self.logit_softcap)
            scores = scores.float()
            chunk_max = scores.amax(dim=-1)
            if running_max is None:
                running_max = chunk_max
                probabilities = torch.exp(scores - running_max.unsqueeze(-1))
                running_sum = probabilities.sum(dim=-1)
                running_context = torch.einsum(
                    "bhts,bsl->bhtl", probabilities, chunk_latent.float()
                )
                return
            new_max = torch.maximum(running_max, chunk_max)
            old_scale = torch.exp(running_max - new_max)
            probabilities = torch.exp(scores - new_max.unsqueeze(-1))
            assert running_sum is not None and running_context is not None
            running_sum = running_sum * old_scale + probabilities.sum(dim=-1)
            running_context = running_context * old_scale.unsqueeze(-1) + torch.einsum(
                "bhts,bsl->bhtl", probabilities, chunk_latent.float()
            )
            running_max = new_max

        for chunk_latent, chunk_rope in previous.iter_chunks(
            dtype=latent.dtype, device=latent.device
        ):
            consume(chunk_latent, chunk_rope)
        # A decoder token attends to itself.
        consume(latent, k_rope)
        assert running_context is not None and running_sum is not None
        context_latent = (running_context / running_sum.clamp_min(1e-20).unsqueeze(-1)).to(q_content.dtype)
        context_latent = context_latent.transpose(1, 2)  # [B,T,H,L]
        return torch.einsum("bthl,hdl->bthd", context_latent, wv)

    def forward(
        self,
        hidden: torch.Tensor,
        position_ids: torch.Tensor,
        cache: LatentLayerCache | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        q_content, q_rope, latent, k_rope, _ = self._project(hidden, position_ids)
        if cache is not None and cache.length and hidden.shape[1] == 1:
            out = self._absorbed_decode(q_content, q_rope, latent, k_rope, cache)
        else:
            out = self._full_attention(q_content, q_rope, latent, k_rope, cache)

        if use_cache and cache is not None:
            cache.append(latent, k_rope, self.window, self.sink_tokens)

        bsz, seq_len = hidden.shape[:2]
        out = out.reshape(bsz, seq_len, self.d_model)
        if self.gate_proj is not None:
            out = out * torch.sigmoid(self.gate_proj(hidden))
        return self.out_proj(out)

    @torch.no_grad()
    def apply_qk_clip(self, tau: float) -> dict[str, float]:
        maxima = self.last_max_logits.float()
        violating = maxima > tau
        count = int(violating.sum().item())
        before = float(maxima.max().item()) if maxima.numel() else 0.0
        if count:
            gamma = torch.ones_like(maxima)
            gamma[violating] = tau / maxima[violating]
            sqrt_gamma = gamma.sqrt()
            q_weight = self.q_up.weight.view(
                self.n_heads, self.head_dim + self.rope_dim, -1
            )
            k_weight = self.k_up.weight.view(self.n_heads, self.head_dim, self.latent_rank)
            q_weight[:, : self.head_dim].mul_(sqrt_gamma[:, None, None].to(q_weight.dtype))
            q_weight[:, self.head_dim :].mul_(gamma[:, None, None].to(q_weight.dtype))
            k_weight.mul_(sqrt_gamma[:, None, None].to(k_weight.dtype))
        self.last_max_logits.zero_()
        return {"heads_clipped": float(count), "max_logit_before": before}
