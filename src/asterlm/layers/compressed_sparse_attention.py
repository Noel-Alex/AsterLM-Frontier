from __future__ import annotations

import torch
from torch import nn

from .deepseek_v4_reference import GatedKVCompressor, compressed_sparse_topk
from .norm import RMSNorm
from .rotary import RotaryEmbedding, apply_rotary


class CompressedSparseAttentionReference(nn.Module):
    """Inspectable no-cache CSA/HCA attention following DeepSeek V4 semantics.

    Ratio four enables learned compressed sparse selection (CSA). Other ratios
    attend to every causally completed compressed summary (HCA). Both retain the
    uncompressed local window and a learnable attention sink. Quantization and the
    production sparse gather kernel are deliberately outside this oracle.
    """

    def __init__(
        self,
        input_dim: int,
        n_heads: int,
        head_dim: int,
        rope_dim: int,
        max_seq_len: int,
        *,
        local_window: int = 128,
        compress_ratio: int = 4,
        index_topk: int = 1024,
        q_lora_rank: int | None = None,
        index_n_heads: int = 64,
        index_head_dim: int = 128,
        output_groups: int = 1,
        output_lora_rank: int | None = None,
        norm_eps: float = 1e-6,
        rope_theta: float = 1_000_000.0,
    ) -> None:
        super().__init__()
        if min(input_dim, n_heads, head_dim, max_seq_len, local_window) <= 0:
            raise ValueError("attention dimensions and window must be positive")
        if rope_dim <= 0 or rope_dim % 2 or rope_dim > head_dim:
            raise ValueError("rope_dim must be positive, even, and at most head_dim")
        if n_heads % output_groups:
            raise ValueError("n_heads must be divisible by output_groups")
        if compress_ratio <= 1:
            raise ValueError("compress_ratio must exceed one")
        if compress_ratio == 4 and (index_n_heads <= 0 or index_head_dim < rope_dim):
            raise ValueError("CSA index geometry is invalid")

        self.input_dim = input_dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.rope_dim = rope_dim
        self.local_window = local_window
        self.compress_ratio = compress_ratio
        self.index_topk = index_topk
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.output_groups = output_groups
        self.output_lora_rank = output_lora_rank or max(1, input_dim // output_groups)
        self.norm_eps = norm_eps

        rank = q_lora_rank or input_dim
        self.q_down = nn.Linear(input_dim, rank, bias=False)
        self.q_norm = RMSNorm(rank, norm_eps)
        self.q_up = nn.Linear(rank, n_heads * head_dim, bias=False)
        self.kv_proj = nn.Linear(input_dim, head_dim, bias=False)
        self.kv_norm = RMSNorm(head_dim, norm_eps)
        self.compressor = GatedKVCompressor(input_dim, head_dim, compress_ratio, eps=norm_eps)
        self.attention_sink = nn.Parameter(torch.zeros(n_heads, dtype=torch.float32))

        heads_per_group = n_heads // output_groups
        self.output_down = nn.Parameter(
            torch.empty(
                output_groups,
                self.output_lora_rank,
                heads_per_group * head_dim,
            )
        )
        self.output_up = nn.Linear(output_groups * self.output_lora_rank, input_dim, bias=False)

        if compress_ratio == 4:
            self.index_q_up: nn.Linear | None = nn.Linear(
                rank, index_n_heads * index_head_dim, bias=False
            )
            self.index_weights: nn.Linear | None = nn.Linear(
                input_dim, index_n_heads, bias=False
            )
            self.index_compressor: GatedKVCompressor | None = GatedKVCompressor(
                input_dim, index_head_dim, compress_ratio, eps=norm_eps
            )
        else:
            self.index_q_up = None
            self.index_weights = None
            self.index_compressor = None

        self.rope = RotaryEmbedding(
            dim=rope_dim,
            theta=rope_theta,
            max_position=max_seq_len,
        )
        self.last_selected_indices: torch.Tensor | None = None
        self.reset_parameters()

    @property
    def is_csa(self) -> bool:
        return self.compress_ratio == 4

    def reset_parameters(self) -> None:
        for module in (self.q_down, self.q_up, self.kv_proj, self.output_up):
            nn.init.xavier_uniform_(module.weight)
        nn.init.xavier_uniform_(self.output_down)
        nn.init.zeros_(self.attention_sink)
        if self.index_q_up is not None:
            nn.init.xavier_uniform_(self.index_q_up.weight)
        if self.index_weights is not None:
            nn.init.xavier_uniform_(self.index_weights.weight)

    def _apply_rope(self, value: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if positions.ndim == 1:
            positions = positions.unsqueeze(0).expand(value.shape[0], -1)
        rope_value = value[..., -self.rope_dim :].unsqueeze(2)
        cos, sin = self.rope(rope_value, positions)
        rotated = apply_rotary(rope_value, cos, sin).squeeze(2)
        return torch.cat((value[..., : -self.rope_dim], rotated), dim=-1)

    def _inverse_rope(self, value: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        rope_value = value[..., -self.rope_dim :]
        cos, sin = self.rope(rope_value, positions)
        rotated = apply_rotary(rope_value, cos, -sin)
        return torch.cat((value[..., : -self.rope_dim], rotated), dim=-1)

    def _local_indices(self, sequence: int, batch: int, device: torch.device) -> torch.Tensor:
        query_positions = torch.arange(sequence, device=device).view(-1, 1)
        offsets = torch.arange(self.local_window - 1, -1, -1, device=device).view(1, -1)
        indices = query_positions - offsets
        indices = torch.where(indices >= 0, indices, indices.new_full((), -1))
        return indices.unsqueeze(0).expand(batch, -1, -1)

    def _dense_compressed_indices(
        self, sequence: int, compressed_count: int, batch: int, device: torch.device
    ) -> torch.Tensor:
        if compressed_count == 0:
            return torch.empty(batch, sequence, 0, dtype=torch.long, device=device)
        positions = torch.arange(sequence, device=device)
        visible = torch.div(
            positions + 1, self.compress_ratio, rounding_mode="floor"
        )
        groups = torch.arange(compressed_count, device=device)
        indices = groups.view(1, -1).expand(sequence, -1)
        indices = torch.where(
            groups.view(1, -1) < visible.view(-1, 1), indices, indices.new_full((), -1)
        )
        return indices.unsqueeze(0).expand(batch, -1, -1)

    def _compressed_indices(
        self,
        hidden: torch.Tensor,
        normalized_query_rank: torch.Tensor,
        positions: torch.Tensor,
        compressed_count: int,
    ) -> torch.Tensor:
        batch, sequence, _ = hidden.shape
        if not self.is_csa:
            return self._dense_compressed_indices(
                sequence, compressed_count, batch, hidden.device
            )
        assert self.index_q_up is not None
        assert self.index_weights is not None
        assert self.index_compressor is not None
        index_query = self.index_q_up(normalized_query_rank).view(
            batch, sequence, self.index_n_heads, self.index_head_dim
        )
        query_rope = index_query[..., -self.rope_dim :]
        cos, sin = self.rope(query_rope, positions)
        index_query = torch.cat(
            (
                index_query[..., : -self.rope_dim],
                apply_rotary(query_rope, cos, sin),
            ),
            dim=-1,
        )
        index_keys = self.index_compressor(hidden)
        group_positions = torch.arange(
            0,
            compressed_count * self.compress_ratio,
            self.compress_ratio,
            device=hidden.device,
        )
        index_keys = self._apply_rope(index_keys, group_positions)
        indices, _ = compressed_sparse_topk(
            index_query,
            index_keys,
            self.index_weights(hidden),
            compress_ratio=self.compress_ratio,
            topk=self.index_topk,
            offset=sequence,
        )
        return indices

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3 or hidden.shape[-1] != self.input_dim:
            raise ValueError(
                f"hidden must have shape [batch, sequence, {self.input_dim}]"
            )
        batch, sequence, _ = hidden.shape
        if sequence == 0:
            return hidden
        positions = torch.arange(sequence, device=hidden.device).unsqueeze(0).expand(batch, -1)

        normalized_query_rank = self.q_norm(self.q_down(hidden))
        query = self.q_up(normalized_query_rank).view(
            batch, sequence, self.n_heads, self.head_dim
        )
        query = query * torch.rsqrt(
            query.float().square().mean(dim=-1, keepdim=True) + self.norm_eps
        ).to(query.dtype)
        query_rope = query[..., -self.rope_dim :]
        cos, sin = self.rope(query_rope, positions)
        query = torch.cat(
            (query[..., : -self.rope_dim], apply_rotary(query_rope, cos, sin)), dim=-1
        )

        local_kv = self._apply_rope(self.kv_norm(self.kv_proj(hidden)), positions)
        compressed_kv = self.compressor(hidden)
        compressed_count = compressed_kv.shape[1]
        group_positions = torch.arange(
            0,
            compressed_count * self.compress_ratio,
            self.compress_ratio,
            device=hidden.device,
        )
        compressed_kv = self._apply_rope(compressed_kv, group_positions)
        all_kv = torch.cat((local_kv, compressed_kv), dim=1)

        local_indices = self._local_indices(sequence, batch, hidden.device)
        compressed_indices = self._compressed_indices(
            hidden, normalized_query_rank, positions, compressed_count
        )
        if not self.is_csa:
            compressed_indices = torch.where(
                compressed_indices >= 0,
                compressed_indices + sequence,
                compressed_indices,
            )
        selected_indices = torch.cat((local_indices, compressed_indices), dim=-1)
        self.last_selected_indices = selected_indices.detach()

        valid = selected_indices >= 0
        safe_indices = selected_indices.clamp_min(0)
        gather_source = all_kv.unsqueeze(1).expand(-1, sequence, -1, -1)
        selected_kv = torch.gather(
            gather_source,
            2,
            safe_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim),
        )
        scores = torch.einsum(
            "bshd,bskd->bshk", query.float(), selected_kv.float()
        ) * self.head_dim**-0.5
        scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
        sink_scores = self.attention_sink.view(1, 1, -1, 1).expand(
            batch, sequence, -1, -1
        )
        weights = torch.cat((scores, sink_scores), dim=-1).softmax(dim=-1)[..., :-1]
        output = torch.einsum("bshk,bskd->bshd", weights, selected_kv.float())
        output = self._inverse_rope(output.to(hidden.dtype), positions)

        heads_per_group = self.n_heads // self.output_groups
        grouped = output.reshape(
            batch,
            sequence,
            self.output_groups,
            heads_per_group * self.head_dim,
        )
        low_rank = torch.einsum(
            "bsgd,grd->bsgr", grouped, self.output_down.to(grouped.dtype)
        )
        return self.output_up(low_rank.flatten(2))
