from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TEGroupedRoutedExperts:
    """Transformer Engine grouped execution bridge for Aster routed experts.

    The existing SwiGLU expert modules remain the authoritative parameter owners.
    This bridge is intentionally not registered as a child module of
    DeepSeekStyleMoE. Before every grouped call it binds TE GroupedLinear weight
    slots to those authoritative expert Parameters, preserving Aster's existing
    state-dict and optimizer parameter layout.
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
        self.dim = dim
        self.expert_hidden = expert_hidden
        self.num_experts = num_experts
        self.dropout = float(dropout)
        self.align = int(align)

        try:
            import transformer_engine.pytorch as te
            from transformer_engine.pytorch.ops import GroupedLinear, Sequential, SwiGLU
        except ImportError as exc:
            raise ImportError(
                "ASTER_MOE_IMPL=grouped requires Transformer Engine with "
                "transformer_engine.pytorch.ops.GroupedLinear support."
            ) from exc

        self.te = te

        # Meta construction avoids allocating a duplicate copy of all routed expert
        # weights. Weight slots are rebound to the authoritative Aster Parameters.
        self.gate_up = GroupedLinear(
            num_groups=num_experts,
            in_features=dim,
            out_features=2 * expert_hidden,
            bias=False,
            dtype=torch.float32,
            device="meta",
        )
        self.down = GroupedLinear(
            num_groups=num_experts,
            in_features=expert_hidden,
            out_features=dim,
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
                raise RuntimeError("Grouped MoE expects SwiGLU experts with gate_up/down projections")
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

        # Rebind in case Module.to()/dtype conversion replaced Parameter objects.
        self._bind_authoritative_weights()

        num_tokens = flat.shape[0]
        routing_map = torch.zeros(
            num_tokens,
            self.num_experts,
            dtype=torch.bool,
            device=flat.device,
        )
        routing_map.scatter_(1, top_idx, True)

        # Preserve Aster's FP32 routing probabilities.
        routing_probs = torch.zeros(
            num_tokens,
            self.num_experts,
            dtype=top_weight.dtype,
            device=flat.device,
        )
        routing_probs.scatter_(1, top_idx, top_weight)

        tokens_per_expert = torch.bincount(
            top_idx.reshape(-1),
            minlength=self.num_experts,
        ).to(torch.int32)

        (
            expert_input,
            permuted_probs,
            row_id_map,
            pad_offsets,
            padded_tokens_per_expert,
        ) = self.te.moe_permute_and_pad_with_probs(
            flat,
            routing_probs,
            routing_map,
            tokens_per_expert,
            self.align,
        )

        split_sizes = padded_tokens_per_expert.to(
            device=flat.device,
            dtype=torch.int32,
        )

        expert_output = self.fused(
            expert_input,
            split_sizes,
            split_sizes,
        )

        if self.dropout:
            expert_output = F.dropout(
                expert_output,
                p=self.dropout,
                training=bool(self.routed_experts[0].training),
            )

        # Fused unpermute removes padding, applies routed probabilities, and
        # accumulates top-k contributions to the original token positions.
        return self.te.moe_unpermute(
            expert_output,
            row_id_map,
            merging_probs=permuted_probs,
            restore_shape=flat.shape,
            map_type="mask",
            pad_offsets=pad_offsets,
        )
