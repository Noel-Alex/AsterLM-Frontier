from __future__ import annotations

import torch
from torch.nn import functional as F

from .ffn import situ_glu
from .moe_grouped_torch import TorchGroupedRoutedExperts


class TorchAOFP8GroupedRoutedExperts(TorchGroupedRoutedExperts):
    """Dropless TorchAO rowwise-FP8 routed experts for Hopper/Blackwell gates.

    Canonical parameters remain BF16 and keep their ordinary Aster checkpoint
    names. Only the two routed-expert grouped GEMMs dynamically quantize their
    operands to FP8. Since TorchAO 0.17 does not yet implement its advertised
    uneven-group padding flag, dispatch is padded on-device to 16 rows per expert
    before calling the differentiable scaled grouped GEMM.
    """

    def __init__(self, *args, align: int = 16, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if align != 16:
            raise ValueError("TorchAO FP8 grouped GEMM requires 16-row alignment")
        try:
            from torchao.prototype.moe_training import (
                _to_fp8_rowwise_then_scaled_grouped_mm,
            )
        except ImportError as exc:
            raise ImportError(
                "moe_implementation=torchao_fp8 requires torchao>=0.17 with "
                "prototype.moe_training support"
            ) from exc
        self.align = int(align)
        self._fp8_grouped_mm = _to_fp8_rowwise_then_scaled_grouped_mm

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

        sorted_expert_ids, order = torch.sort(expert_ids)
        sorted_token_idx = torch.div(order, top_k, rounding_mode="floor")
        sorted_weight = assignment_weight.index_select(0, order)
        counts = torch.bincount(expert_ids, minlength=self.num_experts).to(torch.long)
        padded_counts = ((counts + self.align - 1) // self.align) * self.align
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

        aligned_assignments = (
            (num_assignments + self.align - 1) // self.align
        ) * self.align
        capacity = aligned_assignments + self.num_experts * self.align
        slack = capacity - padded_counts.sum()
        padded_counts = padded_counts.clone()
        padded_counts[-1] += slack
        offsets = padded_counts.cumsum(dim=0, dtype=torch.int32)

        packed = flat.new_zeros((capacity, self.dim))
        packed.index_copy_(0, real_positions, flat.index_select(0, sorted_token_idx))
        return packed, offsets, real_positions, sorted_token_idx, sorted_weight

    def _grouped_mm(
        self,
        activations: torch.Tensor,
        weights: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        # Canonical Linear weights are [experts, out, in]. TorchAO consumes a
        # column-major [experts, in, out] transposed view.
        transposed = weights.transpose(1, 2)
        return self._fp8_grouped_mm(
            activations,
            transposed,
            offsets,
            out_dtype=torch.bfloat16,
            float8_dtype=torch.float8_e4m3fn,
            pad_token_groups_for_grouped_mm=False,
        )

    def forward(
        self,
        flat: torch.Tensor,
        top_idx: torch.Tensor,
        top_weight: torch.Tensor,
    ) -> torch.Tensor:
        if not flat.is_cuda:
            raise RuntimeError("TorchAO FP8 grouped MoE is a CUDA-only backend")
        capability = torch.cuda.get_device_capability(flat.device)
        if capability not in {(9, 0), (10, 0)}:
            raise RuntimeError(
                "TorchAO FP8 grouped MoE requires SM90 or SM100 because PyTorch "
                f"does not implement torch._scaled_grouped_mm on SM{capability[0]}{capability[1]}"
            )
        if flat.dtype != torch.bfloat16:
            raise RuntimeError("TorchAO FP8 grouped MoE requires BF16 master activations")
        if flat.ndim != 2 or flat.shape[-1] != self.dim:
            raise ValueError(f"Expected routed input [N, {self.dim}], got {tuple(flat.shape)}")
        if top_idx.ndim != 2 or top_weight.shape != top_idx.shape:
            raise ValueError("top_idx and top_weight must have matching [N, top_k] shapes")

        self.forward_calls += 1
        packed, offsets, real_positions, sorted_token_idx, sorted_weight = (
            self._dispatch_and_pad(flat, top_idx, top_weight)
        )
        gate_up_weights = self._stack_parameters(
            "gate_up", [expert.gate_up.weight for expert in self.routed_experts]
        )
        gate_up = self._grouped_mm(packed, gate_up_weights, offsets)
        gate, up = gate_up.chunk(2, dim=-1)
        hidden = (
            situ_glu(gate, up, self.beta_gate, self.beta_up)
            if self.activation_name == "situ_glu"
            else F.silu(gate) * up
        )
        down_weights = self._stack_parameters(
            "down", [expert.down.weight for expert in self.routed_experts]
        )
        padded_output = self._grouped_mm(hidden.contiguous(), down_weights, offsets)
        if self.dropout:
            padded_output = F.dropout(
                padded_output,
                p=self.dropout,
                training=bool(self.routed_experts[0].training),
            )

        expert_output = padded_output.index_select(0, real_positions)
        weighted = expert_output * sorted_weight.to(expert_output.dtype).unsqueeze(-1)
        routed_out = torch.zeros_like(flat)
        routed_out.index_add_(0, sorted_token_idx, weighted)
        return routed_out
