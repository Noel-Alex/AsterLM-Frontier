from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TEGroupedRoutedExperts:
    """Transformer Engine grouped execution bridge for Aster routed experts.

    Aster's existing ``self.routed`` SwiGLU modules remain the authoritative
    parameter owners. This object is intentionally not an ``nn.Module`` child of
    ``DeepSeekStyleMoE`` so checkpoint/state-dict/optimizer ownership remains
    unchanged.

    Ada + delayed-scaling FP8 needs each expert's grouped-GEMM M dimension
    aligned to 16. The dispatcher below performs that padding with tensor
    indexing only, then removes dummy rows before weighted top-k accumulation.
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
                "ASTER_MOE_IMPL=grouped requires Transformer Engine with "
                "transformer_engine.pytorch.ops.GroupedLinear support."
            ) from exc

        # Meta construction avoids allocating a duplicate set of expert weights.
        # All registered weight slots are rebound to the already-existing Aster
        # expert Parameters before execution.
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
                raise RuntimeError(
                    "Grouped MoE expects SwiGLU experts with gate_up/down projections"
                )
            if not hasattr(gate_up, "weight") or not hasattr(down, "weight"):
                raise RuntimeError(
                    "Grouped MoE requires explicit expert weight Parameters"
                )
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
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Pack top-k assignments by expert and pad each expert to ``self.align``.

        Returns:
            packed:
                Padded expert-sorted input rows.
            split_sizes:
                Per-expert padded row counts for TE GroupedLinear.
            real_positions:
                Positions of real (non-padding) rows inside ``packed``.
            sorted_token_idx:
                Original token index for each real expert-sorted row.
            sorted_weight:
                Routing weight corresponding to each real expert-sorted row.
        """
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

        # Sort all top-k assignments once so every expert occupies one contiguous
        # segment. This replaces the reference path's eight torch.where launches.
        order = torch.argsort(expert_ids)
        sorted_expert_ids = expert_ids.index_select(0, order)
        sorted_token_idx = assignment_token_idx.index_select(0, order)
        sorted_weight = assignment_weight.index_select(0, order)

        counts = torch.bincount(
            expert_ids,
            minlength=self.num_experts,
        ).to(torch.long)

        padded_counts = ((counts + self.align - 1) // self.align) * self.align

        # Give an empty expert one legal all-zero tile. Healthy Aster routing
        # normally makes this irrelevant, but it avoids backend-specific behavior
        # for zero-sized grouped GEMMs and keeps the path fail-safe.
        padded_counts = torch.where(
            padded_counts > 0,
            padded_counts,
            torch.full_like(padded_counts, self.align),
        )

        padding_per_expert = padded_counts - counts
        padding_before = (
            torch.cumsum(padding_per_expert, dim=0) - padding_per_expert
        )

        # Real rows keep their expert-sorted order; cumulative padding shifts each
        # expert's segment to its aligned location.
        real_positions = (
            torch.arange(num_assignments, device=flat.device, dtype=torch.long)
            + padding_before.index_select(0, sorted_expert_ids)
        )

        # A scalar conversion is required to size the packed buffer. This is one
        # host synchronization per MoE invocation, substantially less CPU launch
        # work than the original per-expert GEMM loop. We can remove this sync in
        # a later custom-dispatch kernel if profiling says it matters.
        total_padded = int(padded_counts.sum().item())

        gathered = flat.index_select(0, sorted_token_idx)
        packed = flat.new_zeros((total_padded, self.dim))
        packed = packed.index_copy(0, real_positions, gathered)

        split_sizes = padded_counts.to(
            device=flat.device,
            dtype=torch.int32,
        )

        return (
            packed,
            split_sizes,
            real_positions,
            sorted_token_idx,
            sorted_weight,
        )

    def forward(
        self,
        flat: torch.Tensor,
        top_idx: torch.Tensor,
        top_weight: torch.Tensor,
    ) -> torch.Tensor:
        if flat.ndim != 2 or flat.shape[-1] != self.dim:
            raise ValueError(
                f"Expected routed input [N, {self.dim}], got {tuple(flat.shape)}"
            )
        if top_idx.ndim != 2 or top_weight.shape != top_idx.shape:
            raise ValueError(
                "top_idx and top_weight must have matching [N, top_k] shapes"
            )

        # Module.to()/dtype conversion may replace the authoritative Parameters,
        # so refresh the GroupedLinear aliases immediately before execution.
        self._bind_authoritative_weights()

        (
            packed,
            split_sizes,
            real_positions,
            sorted_token_idx,
            sorted_weight,
        ) = self._dispatch_and_pad(flat, top_idx, top_weight)

        expert_output_padded = self.fused(
            packed,
            split_sizes,
            split_sizes,
        )

        if self.dropout:
            expert_output_padded = F.dropout(
                expert_output_padded,
                p=self.dropout,
                training=bool(self.routed_experts[0].training),
            )

        # Select only real rows before routing-weight application. Dummy rows
        # therefore have zero upstream gradient and cannot change expert wgrads.
        expert_output = expert_output_padded.index_select(
            0,
            real_positions,
        )
        weighted = expert_output * sorted_weight.to(
            expert_output.dtype
        ).unsqueeze(-1)

        # Restore original token order and sum top-k expert contributions.
        routed_out = torch.zeros_like(flat)
        routed_out.index_add_(0, sorted_token_idx, weighted)
        return routed_out
