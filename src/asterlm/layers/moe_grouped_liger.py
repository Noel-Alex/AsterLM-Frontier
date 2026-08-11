from __future__ import annotations

import torch
from torch import nn

from .moe_grouped_cutlass import _CachedParameterStack


class LigerGroupedRoutedExperts:
    """Dropless fused SwiGLU experts using Liger's Triton training kernel.

    Canonical expert Parameters remain owned by Aster's ordinary expert modules so
    checkpoints and optimizer partitioning are backend-independent. The packed
    tensors use the same versioned cache/gradient bridge as the CUTLASS path.

    Liger 0.8.x fuses ordinary SwiGLU. Kimi K3's SiTU-GLU has different semantics
    and is rejected rather than silently approximated.
    """

    def __init__(
        self,
        routed_experts: nn.ModuleList,
        dim: int,
        expert_hidden: int,
        num_experts: int,
        dropout: float = 0.0,
    ) -> None:
        if len(routed_experts) != num_experts:
            raise RuntimeError(
                f"Expected {num_experts} routed experts, got {len(routed_experts)}"
            )
        if dropout:
            raise ValueError("LigerExperts does not implement Aster expert dropout")
        try:
            from liger_kernel.ops.fused_moe import LigerFusedMoEFunction
        except ImportError as exc:
            raise ImportError(
                "moe_implementation=liger requires liger-kernel>=0.8,<0.9"
            ) from exc

        self.routed_experts = routed_experts
        self.dim = int(dim)
        self.expert_hidden = int(expert_hidden)
        self.num_experts = int(num_experts)
        self.function = LigerFusedMoEFunction
        self._weight_cache: dict[str, tuple[tuple[int, ...], torch.Tensor]] = {}
        self.cache_hits = 0
        self.cache_refreshes = 0

        for expert in self.routed_experts:
            gate_up = getattr(expert, "gate_up", None)
            down = getattr(expert, "down", None)
            if gate_up is None or down is None:
                raise RuntimeError("Liger grouped MoE expects gate_up/down modules")
            if not hasattr(gate_up, "weight") or not hasattr(down, "weight"):
                raise RuntimeError("Liger grouped MoE requires explicit expert weights")
            if getattr(expert, "activation_name", "swiglu") != "swiglu":
                raise ValueError(
                    "LigerExperts 0.8 supports SwiGLU only; SiTU-GLU requires a "
                    "separately parity-tested kernel"
                )

    def _stack_parameters(
        self, cache_key: str, parameters: list[torch.Tensor]
    ) -> torch.Tensor:
        versions = tuple(parameter._version for parameter in parameters)
        cached_entry = self._weight_cache.get(cache_key)
        if cached_entry is None or cached_entry[0] != versions:
            with torch.no_grad():
                cached = torch.stack(
                    [parameter.detach() for parameter in parameters], dim=0
                ).contiguous()
            self._weight_cache[cache_key] = (versions, cached)
            self.cache_refreshes += 1
        else:
            cached = cached_entry[1]
            self.cache_hits += 1
        return _CachedParameterStack.apply(cached, *parameters)

    def __call__(
        self,
        flat: torch.Tensor,
        top_idx: torch.Tensor,
        top_weight: torch.Tensor,
    ) -> torch.Tensor:
        if not flat.is_cuda:
            raise RuntimeError("Liger grouped MoE is a CUDA-only execution backend")
        if flat.ndim != 2 or flat.shape[-1] != self.dim:
            raise ValueError(f"Expected routed input [N, {self.dim}], got {tuple(flat.shape)}")
        if top_idx.ndim != 2 or top_weight.shape != top_idx.shape:
            raise ValueError("top_idx and top_weight must have matching [N, top_k] shapes")

        gate_up = self._stack_parameters(
            "gate_up", [expert.gate_up.weight for expert in self.routed_experts]
        )
        down = self._stack_parameters(
            "down", [expert.down.weight for expert in self.routed_experts]
        )
        return self.function.apply(
            flat.contiguous(),
            gate_up,
            down,
            top_idx.to(dtype=torch.int32).contiguous(),
            top_weight.float().contiguous(),
        ).to(dtype=flat.dtype)
