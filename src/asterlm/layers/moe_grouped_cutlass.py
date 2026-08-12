from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .ffn import situ_glu


def pack_parameter_storage(parameters: list[nn.Parameter]) -> torch.Tensor:
    """Rebind equal-shaped Parameters to disjoint views of one contiguous tensor."""

    if not parameters:
        raise ValueError("Cannot pack an empty parameter list")
    shape = parameters[0].shape
    device = parameters[0].device
    dtype = parameters[0].dtype
    if any(
        parameter.shape != shape
        or parameter.device != device
        or parameter.dtype != dtype
        for parameter in parameters
    ):
        raise ValueError("Packed expert Parameters must share shape, device, and dtype")
    with torch.no_grad():
        packed = torch.stack([parameter.detach() for parameter in parameters], dim=0)
        for index, parameter in enumerate(parameters):
            parameter.data = packed[index]
    return packed


def materialize_parameter_storage(parameters: list[nn.Parameter]) -> int:
    """Give packed parameter views independent storage for serialization."""

    bytes_materialized = 0
    with torch.no_grad():
        for parameter in parameters:
            independent = parameter.detach().clone()
            bytes_materialized += independent.numel() * independent.element_size()
            parameter.data = independent
    return bytes_materialized


class _CachedParameterStack(torch.autograd.Function):
    """Expose a detached packed cache while returning gradients to source Parameters."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        cached: torch.Tensor,
        *parameters: torch.Tensor,
    ) -> torch.Tensor:
        ctx.parameter_count = len(parameters)
        return cached

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[None, *tuple[torch.Tensor, ...]]:
        gradients = tuple(grad_output.unbind(0))
        if len(gradients) != ctx.parameter_count:
            raise RuntimeError("Packed expert gradient count does not match source Parameters")
        return (None, *gradients)


class CUTLASSGroupedRoutedExperts:
    """Dropless grouped SwiGLU using the MegaBlocks CUTLASS operators.

    This is an experimental Ada-oriented bridge. The existing expert modules keep
    ownership of the canonical Parameters and checkpoint names. After device/dtype
    placement their disjoint parameter views can share one contiguous backing tensor,
    so grouped GEMMs do not rematerialize every expert matrix after optimizer steps.

    The upstream permute/unpermute operators keep routing and the weighted top-k
    combine on device. CUTLASS grouped GEMM currently accepts host batch sizes, so
    this path performs one vector device-to-host transfer per MoE invocation rather
    than Transformer Engine's repeated scalar extraction. No assignment is dropped.
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
        try:
            from grouped_gemm import ops
        except ImportError as exc:
            raise ImportError(
                "moe_implementation=cutlass requires the Apache-2.0 nv_grouped_gemm "
                "extension built for the exact PyTorch, CUDA and SM backend"
            ) from exc

        self.routed_experts = routed_experts
        self.dim = int(dim)
        self.expert_hidden = int(expert_hidden)
        self.num_experts = int(num_experts)
        self.dropout = float(dropout)
        self.ops = ops
        self._weight_cache: dict[
            str, tuple[tuple[int, ...], torch.Tensor]
        ] = {}
        self._packed_storage: dict[str, torch.Tensor] = {}
        self.cache_hits = 0
        self.cache_refreshes = 0
        self.cache_refresh_bytes = 0
        self.forward_calls = 0
        self.host_metadata_syncs = 0
        self.storage_pack_count = 0
        self.storage_pack_bytes = 0
        self.storage_materialization_count = 0
        self.storage_materialization_bytes = 0

        for expert in self.routed_experts:
            gate_up = getattr(expert, "gate_up", None)
            down = getattr(expert, "down", None)
            if gate_up is None or down is None:
                raise RuntimeError("CUTLASS grouped MoE expects SwiGLU gate_up/down modules")
            if not hasattr(gate_up, "weight") or not hasattr(down, "weight"):
                raise RuntimeError("CUTLASS grouped MoE requires explicit expert weights")
        self.activation_name = str(
            getattr(self.routed_experts[0], "activation_name", "swiglu")
        )
        if any(
            getattr(expert, "activation_name", "swiglu") != self.activation_name
            for expert in self.routed_experts
        ):
            raise RuntimeError("All grouped experts must use the same activation")
        self.beta_gate = float(getattr(self.routed_experts[0], "beta_gate", 4.0))
        self.beta_up = float(getattr(self.routed_experts[0], "beta_up", 25.0))

    def _stack_parameters(
        self,
        cache_key: str,
        parameters: list[torch.Tensor],
    ) -> torch.Tensor:
        versions = tuple(parameter._version for parameter in parameters)
        cached_entry = self._weight_cache.get(cache_key)
        if cached_entry is None or cached_entry[0] != versions:
            packed_storage = self._packed_storage.get(cache_key)
            if packed_storage is not None:
                cached = packed_storage
            else:
                with torch.no_grad():
                    cached = torch.stack(
                        [parameter.detach() for parameter in parameters], dim=0
                    ).contiguous()
                self.cache_refresh_bytes += cached.numel() * cached.element_size()
            self._weight_cache[cache_key] = (versions, cached)
            self.cache_refreshes += 1
        else:
            cached = cached_entry[1]
            self.cache_hits += 1
        return _CachedParameterStack.apply(cached, *parameters)

    def pack_parameter_storage(self) -> dict[str, float]:
        """Pack canonical expert Parameters after device/dtype placement."""

        groups = {
            "gate_up": [expert.gate_up.weight for expert in self.routed_experts],
            "down": [expert.down.weight for expert in self.routed_experts],
        }
        self._weight_cache.clear()
        self._packed_storage.clear()
        bytes_packed = 0
        for name, parameters in groups.items():
            packed = pack_parameter_storage(parameters)
            self._packed_storage[name] = packed
            bytes_packed += packed.numel() * packed.element_size()
        self.storage_pack_count += 1
        self.storage_pack_bytes += bytes_packed
        return {
            "moe_backend_storage_pack_count": float(self.storage_pack_count),
            "moe_backend_storage_pack_gib": self.storage_pack_bytes / (2**30),
        }

    def materialize_parameter_storage(self) -> dict[str, float]:
        """Temporarily remove shared partial-storage views for safe serialization."""

        if not self._packed_storage:
            return {}
        groups = {
            "gate_up": [expert.gate_up.weight for expert in self.routed_experts],
            "down": [expert.down.weight for expert in self.routed_experts],
        }
        bytes_materialized = 0
        materialized_groups = 0
        self._weight_cache.clear()
        for name, parameters in groups.items():
            if name not in self._packed_storage:
                continue
            # Retain the backing tensor until every view in this group has been
            # cloned, then release it before materializing the next group.
            backing = self._packed_storage.pop(name)
            bytes_materialized += materialize_parameter_storage(parameters)
            del backing
            materialized_groups += 1
        self.storage_materialization_count += 1
        self.storage_materialization_bytes += bytes_materialized
        return {
            "moe_backend_storage_materialization_count": float(
                self.storage_materialization_count
            ),
            "moe_backend_storage_materialization_groups": float(materialized_groups),
            "moe_backend_storage_materialization_gib": (
                self.storage_materialization_bytes / (2**30)
            ),
        }

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
        if not flat.is_cuda:
            raise RuntimeError("CUTLASS grouped MoE is a CUDA-only execution backend")
        if flat.ndim != 2 or flat.shape[-1] != self.dim:
            raise ValueError(f"Expected routed input [N, {self.dim}], got {tuple(flat.shape)}")
        if top_idx.ndim != 2 or top_weight.shape != top_idx.shape:
            raise ValueError("top_idx and top_weight must have matching [N, top_k] shapes")

        self.forward_calls += 1
        route = top_idx.to(dtype=torch.int32).contiguous()
        # The grouped GEMM launcher consumes CPU sizes. This is one bulk vector
        # synchronization; keep it explicit so profiles cannot hide the cost.
        batch_sizes = torch.bincount(
            route.reshape(-1), minlength=self.num_experts
        ).to(device="cpu", dtype=torch.long)
        self.host_metadata_syncs += 1
        permuted, row_id_map = self.ops.permute(
            flat.contiguous(), route, max_token_num=flat.shape[0]
        )

        gate_up_weights = self._stack_parameters(
            "gate_up",
            [expert.gate_up.weight for expert in self.routed_experts],
        )
        gate_up = self.ops.gmm(
            permuted, gate_up_weights, batch_sizes, trans_b=True
        )
        gate, up = gate_up.chunk(2, dim=-1)
        if self.activation_name == "situ_glu":
            hidden = situ_glu(gate, up, self.beta_gate, self.beta_up)
        else:
            hidden = F.silu(gate) * up

        down_weights = self._stack_parameters(
            "down",
            [expert.down.weight for expert in self.routed_experts],
        )
        expert_output = self.ops.gmm(
            hidden.contiguous(), down_weights, batch_sizes, trans_b=True
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

    def diagnostics(self) -> dict[str, float]:
        return {
            "moe_backend_forward_calls": float(self.forward_calls),
            "moe_backend_host_metadata_syncs": float(self.host_metadata_syncs),
            "moe_backend_weight_cache_refreshes": float(self.cache_refreshes),
            "moe_backend_weight_cache_hits": float(self.cache_hits),
            "moe_backend_weight_cache_refresh_gib": self.cache_refresh_bytes / (2**30),
            "moe_backend_storage_pack_count": float(self.storage_pack_count),
            "moe_backend_storage_pack_gib": self.storage_pack_bytes / (2**30),
            "moe_backend_storage_materialization_count": float(
                self.storage_materialization_count
            ),
            "moe_backend_storage_materialization_gib": (
                self.storage_materialization_bytes / (2**30)
            ),
        }
