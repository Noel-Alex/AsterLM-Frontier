from __future__ import annotations

import os

import torch
from torch import nn
from torch.nn import functional as F


class TEGroupedRoutedExperts:
    """Transformer Engine grouped execution bridge for routed experts.

    Existing expert modules remain the authoritative parameter owners. This bridge is
    intentionally not registered as a child module by its caller, preserving Aster's
    optimizer and checkpoint parameter names.

    Ada FP8 grouped wgrad requires the expert M dimension to be divisible by 16. Each
    expert segment is padded accordingly. Unlike the previous Aster implementation,
    this version never calls ``Tensor.item()`` to size its packed buffer, eliminating a
    GPU->CPU synchronization from every MoE invocation.
    """

    def __init__(
        self,
        routed_experts: nn.ModuleList,
        dim: int,
        expert_hidden: int,
        num_experts: int,
        dropout: float = 0.0,
        align: int = 16,
    ) -> None:
        if align <= 0:
            raise ValueError("Grouped MoE alignment must be positive")
        self.routed_experts = routed_experts
        self.dim = int(dim)
        self.expert_hidden = int(expert_hidden)
        self.num_experts = int(num_experts)
        self.dropout = float(dropout)
        self.align = int(align)
        self.dispatch_impl = os.environ.get("ASTER_MOE_DISPATCH", "current").strip().lower()
        if self.dispatch_impl not in {"current", "te_mask_pad"}:
            raise ValueError(
                "ASTER_MOE_DISPATCH must be either 'current' or 'te_mask_pad', "
                f"got {self.dispatch_impl!r}"
            )

        try:
            from transformer_engine.pytorch.ops import GroupedLinear, Sequential, SwiGLU
        except ImportError as exc:
            raise ImportError(
                "Grouped MoE requires Transformer Engine with "
                "transformer_engine.pytorch.ops.GroupedLinear support."
            ) from exc

        self.gate_up = GroupedLinear(
            num_groups=self.num_experts,
            in_features=self.dim,
            out_features=2 * self.expert_hidden,
            bias=False,
            dtype=torch.float32,
            device="meta",
        )
        self.down = GroupedLinear(
            num_groups=self.num_experts,
            in_features=self.expert_hidden,
            out_features=self.dim,
            bias=False,
            dtype=torch.float32,
            device="meta",
        )
        self.fused = Sequential(self.gate_up, SwiGLU(), self.down)
        self._bind_authoritative_weights()

    def _bind_authoritative_weights(self) -> None:
        if len(self.routed_experts) != self.num_experts:
            raise RuntimeError(
                f"Expected {self.num_experts} routed experts, got {len(self.routed_experts)}"
            )
        for idx, expert in enumerate(self.routed_experts):
            gate_up = getattr(expert, "gate_up", None)
            down = getattr(expert, "down", None)
            if gate_up is None or down is None:
                raise RuntimeError("Grouped MoE expects SwiGLU experts with gate_up/down")
            if not hasattr(gate_up, "weight") or not hasattr(down, "weight"):
                raise RuntimeError("Grouped MoE requires explicit expert weight Parameters")
            setattr(self.gate_up, f"weight{idx}", gate_up.weight)
            setattr(self.down, f"weight{idx}", down.weight)

    def __call__(
        self,
        flat: torch.Tensor,
        top_idx: torch.Tensor,
        top_weight: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward(flat, top_idx, top_weight)

    def _dispatch_and_pad(
        self,
        flat: torch.Tensor,
        top_idx: torch.Tensor,
        top_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_tokens = flat.shape[0]
        top_k = top_idx.shape[1]
        num_assignments = num_tokens * top_k

        expert_ids = top_idx.reshape(-1).to(torch.long)
        assignment_weight = top_weight.reshape(-1)

        # Sort once and derive token indices from the token-major flattened route
        # position. The older path allocated an expanded [tokens, top_k] index tensor
        # and then gathered from it on every MoE layer.
        sorted_expert_ids, order = torch.sort(expert_ids)
        sorted_token_idx = torch.div(order, top_k, rounding_mode="floor")
        sorted_weight = assignment_weight.index_select(0, order)

        counts = torch.bincount(expert_ids, minlength=self.num_experts).to(torch.long)
        padded_counts = ((counts + self.align - 1) // self.align) * self.align
        # Empty experts still receive one legal zero tile. This makes the grouped
        # kernel path well-defined even under pathological early routing collapse.
        padded_counts = torch.where(
            padded_counts > 0,
            padded_counts,
            torch.full_like(padded_counts, self.align),
        )
        padding_per_expert = padded_counts - counts
        padding_before = torch.cumsum(padding_per_expert, dim=0) - padding_per_expert
        real_positions = (
            torch.arange(num_assignments, device=flat.device, dtype=torch.long)
            + padding_before.index_select(0, sorted_expert_ids)
        )

        # Fixed, shape-derived capacity. Worst-case padding is <= align rows per
        # expert. Keep the capacity itself aligned and place any remaining slack into
        # the final expert's zero segment. No device scalar is copied to the host.
        aligned_assignments = (
            (num_assignments + self.align - 1) // self.align
        ) * self.align
        capacity = aligned_assignments + self.num_experts * self.align
        slack = capacity - padded_counts.sum()
        split_sizes_long = padded_counts.clone()
        split_sizes_long[-1] = split_sizes_long[-1] + slack

        gathered = flat.index_select(0, sorted_token_idx)
        packed = flat.new_zeros((capacity, self.dim))
        packed.index_copy_(0, real_positions, gathered)
        split_sizes = split_sizes_long.to(dtype=torch.int32)

        return packed, split_sizes, real_positions, sorted_token_idx, sorted_weight

    def forward(
        self,
        flat: torch.Tensor,
        top_idx: torch.Tensor,
        top_weight: torch.Tensor,
    ) -> torch.Tensor:
        if flat.ndim != 2 or flat.shape[-1] != self.dim:
            raise ValueError(f"Expected routed input [N, {self.dim}], got {tuple(flat.shape)}")
        if top_idx.ndim != 2 or top_weight.shape != top_idx.shape:
            raise ValueError("top_idx and top_weight must have matching [N, top_k] shapes")

        if self.dispatch_impl == "te_mask_pad":
            return self._forward_te_mask_pad(flat, top_idx, top_weight)

        packed, split_sizes, real_positions, sorted_token_idx, sorted_weight = self._dispatch_and_pad(
            flat, top_idx, top_weight
        )
        expert_output_padded = self.fused(packed, split_sizes, split_sizes)
        if self.dropout:
            expert_output_padded = F.dropout(
                expert_output_padded,
                p=self.dropout,
                training=bool(self.routed_experts[0].training),
            )

        expert_output = expert_output_padded.index_select(0, real_positions)
        weighted = expert_output * sorted_weight.to(expert_output.dtype).unsqueeze(-1)
        routed_out = torch.zeros_like(flat)
        routed_out.index_add_(0, sorted_token_idx, weighted)
        return routed_out

    def _forward_te_mask_pad(
        self,
        flat: torch.Tensor,
        top_idx: torch.Tensor,
        top_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Experimental official TE fused permutation/padding and weighted combine."""

        try:
            from transformer_engine.pytorch import moe_permute_and_pad_with_probs, moe_unpermute
        except ImportError as exc:
            raise ImportError(
                "ASTER_MOE_DISPATCH=te_mask_pad requires Transformer Engine MoE permutation ops"
            ) from exc

        routing_map = torch.zeros(
            (flat.shape[0], self.num_experts),
            dtype=torch.int32,
            device=flat.device,
        ).scatter_(1, top_idx, 1)
        dense_probs = torch.zeros(
            (flat.shape[0], self.num_experts),
            dtype=torch.float32,
            device=flat.device,
        ).scatter(1, top_idx, top_weight.float())
        tokens_per_expert = torch.bincount(
            top_idx.reshape(-1),
            minlength=self.num_experts,
        )
        permuted, _, row_id_map, pad_offsets, target_counts = moe_permute_and_pad_with_probs(
            flat,
            dense_probs,
            routing_map,
            tokens_per_expert,
            self.align,
        )
        split_sizes = target_counts.to(torch.int32)
        expert_output = self.fused(permuted, split_sizes, split_sizes)
        if self.dropout:
            expert_output = F.dropout(
                expert_output,
                p=self.dropout,
                training=bool(self.routed_experts[0].training),
            )
        return moe_unpermute(
            expert_output,
            row_id_map,
            merging_probs=dense_probs,
            restore_shape=flat.shape,
            map_type="mask",
            pad_offsets=pad_offsets,
        )
