from __future__ import annotations

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
        assignment_token_idx = (
            torch.arange(num_tokens, device=flat.device, dtype=torch.long)
            .unsqueeze(1)
            .expand(num_tokens, top_k)
            .reshape(-1)
        )
        assignment_weight = top_weight.reshape(-1)

        order = torch.argsort(expert_ids)
        sorted_expert_ids = expert_ids.index_select(0, order)
        sorted_token_idx = assignment_token_idx.index_select(0, order)
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
        capacity_tensor = torch.as_tensor(capacity, device=flat.device, dtype=torch.long)
        slack = capacity_tensor - padded_counts.sum()
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

        self._bind_authoritative_weights()
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
