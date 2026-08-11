from __future__ import annotations

import torch
from torch.nn import functional as F

from .ffn import situ_glu
from .moe_grouped_cutlass import CUTLASSGroupedRoutedExperts
from .routing import fixed_bincount


class TorchGroupedRoutedExperts(CUTLASSGroupedRoutedExperts):
    """GPU-resident grouped experts using native PyTorch grouped MM.

    The routing permutation comes from the same numerically checked grouped-GEMM
    extension as the CUTLASS control, but expert counts remain on GPU and are
    converted to cumulative ``int32`` offsets for ``torch._grouped_mm``. This
    removes the host metadata copy and stream synchronization required by the
    older extension's grouped-GEMM launcher while retaining canonical per-expert
    parameters and checkpoint names.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not hasattr(F, "grouped_mm"):
            raise RuntimeError(
                "ASTER_MOE_IMPL=torch_grouped requires a PyTorch build with "
                "differentiable torch.nn.functional.grouped_mm"
            )

    @staticmethod
    def _grouped_mm(
        activations: torch.Tensor,
        weights: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        # Canonical Linear weights are [experts, out, in]. grouped_mm consumes
        # [experts, in, out]; the transposed view is the supported physical layout.
        return F.grouped_mm(
            activations,
            weights.transpose(1, 2),
            offs=offsets,
        )

    def forward(
        self,
        flat: torch.Tensor,
        top_idx: torch.Tensor,
        top_weight: torch.Tensor,
    ) -> torch.Tensor:
        if not flat.is_cuda:
            raise RuntimeError("PyTorch grouped MoE is a CUDA-only execution backend")
        if flat.dtype != torch.bfloat16:
            raise RuntimeError("PyTorch grouped MoE currently requires BF16 activations")
        if flat.ndim != 2 or flat.shape[-1] != self.dim:
            raise ValueError(f"Expected routed input [N, {self.dim}], got {tuple(flat.shape)}")
        if top_idx.ndim != 2 or top_weight.shape != top_idx.shape:
            raise ValueError("top_idx and top_weight must have matching [N, top_k] shapes")

        route = top_idx.to(dtype=torch.int32).contiguous()
        counts = fixed_bincount(route, self.num_experts, dtype=torch.int32)
        offsets = counts.cumsum(dim=0, dtype=torch.int32)
        permuted, row_id_map = self.ops.permute(
            flat.contiguous(), route, max_token_num=flat.shape[0]
        )

        gate_up_weights = self._stack_parameters(
            "gate_up",
            [expert.gate_up.weight for expert in self.routed_experts],
        )
        gate_up = self._grouped_mm(permuted, gate_up_weights, offsets)
        gate, up = gate_up.chunk(2, dim=-1)
        if self.activation_name == "situ_glu":
            hidden = situ_glu(gate, up, self.beta_gate, self.beta_up)
        else:
            hidden = F.silu(gate) * up

        down_weights = self._stack_parameters(
            "down",
            [expert.down.weight for expert in self.routed_experts],
        )
        expert_output = self._grouped_mm(
            hidden.contiguous(), down_weights, offsets
        )
        if self.dropout:
            expert_output = F.dropout(
                expert_output,
                p=self.dropout,
                training=bool(self.routed_experts[0].training),
            )

        return self.ops.unpermute(
            expert_output.contiguous(),
            row_id_map,
            top_weight.float().contiguous(),
        ).to(dtype=flat.dtype)
