from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .ffn import SwiGLU
from .linear import build_linear
from .moe_grouped_cutlass import CUTLASSGroupedRoutedExperts
from .moe_grouped_liger import LigerGroupedRoutedExperts
from .moe_grouped_te import TEGroupedRoutedExperts
from .moe_grouped_torch import TorchGroupedRoutedExperts
from .moe_grouped_torchao_fp8 import TorchAOFP8GroupedRoutedExperts
from .routing import fixed_bincount


class DeepSeekStyleMoE(nn.Module):
    """Single-GPU, quality-first sparse FFN.

    It combines always-on shared experts with fine-grained routed experts. Routing is
    top-k without token dropping; this is slower than fused grouped GEMMs but avoids
    capacity-loss artifacts and is a dependable reference implementation for ablations.
    """

    def __init__(
        self,
        dim: int,
        expert_hidden: int,
        num_experts: int,
        top_k: int,
        shared_experts: int = 1,
        dropout: float = 0.0,
        router_score: str = "sigmoid",
        balance_strategy: str = "bias",
        bias_update_speed: float = 0.001,
        linear_backend: str = "torch",
        loqt_rank: int = 32,
        loqt_alpha: float = 32.0,
        loqt_group_size: int = 64,
        init_std: float = 0.02,
        moe_impl: str = "reference",
    ) -> None:
        super().__init__()
        if num_experts < 1 or not 1 <= top_k <= num_experts:
            raise ValueError("MoE requires 1 <= top_k <= num_experts")
        if shared_experts < 0:
            raise ValueError("shared_experts must be non-negative")
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.router_score = router_score
        self.balance_strategy = balance_strategy
        self.bias_update_speed = bias_update_speed
        self.linear_backend = linear_backend
        router_backend = "torch" if linear_backend == "transformer_engine" else linear_backend
        self.router = build_linear(dim, num_experts, bias=False, backend=router_backend)
        self.register_buffer("routing_bias", torch.zeros(num_experts, dtype=torch.float32))
        self.register_buffer("load_accumulator", torch.zeros(num_experts, dtype=torch.float32), persistent=False)
        self.register_buffer("load_batches", torch.zeros((), dtype=torch.float32), persistent=False)
        ffn_kwargs = {
            "loqt_rank": loqt_rank,
            "loqt_alpha": loqt_alpha,
            "loqt_group_size": loqt_group_size,
            "init_std": init_std,
        }
        self.routed = nn.ModuleList(
            [SwiGLU(dim, expert_hidden, dropout, linear_backend, **ffn_kwargs) for _ in range(num_experts)]
        )
        self.shared = nn.ModuleList(
            [SwiGLU(dim, expert_hidden, dropout, linear_backend, **ffn_kwargs) for _ in range(shared_experts)]
        )
        requested_impl = moe_impl.strip().lower()
        if requested_impl not in {
            "reference",
            "grouped",
            "cutlass",
            "torch_grouped",
            "torchao_fp8",
            "liger",
        }:
            raise ValueError(
                "moe_impl must be 'reference', 'grouped', 'cutlass', or "
                "'torch_grouped', 'torchao_fp8', or 'liger', "
                f"got {requested_impl!r}"
            )
        if requested_impl == "grouped" and linear_backend != "transformer_engine":
            raise ValueError(
                "moe_impl=grouped currently requires linear_backend='transformer_engine'"
            )
        self.moe_impl = requested_impl
        self._grouped_routed = None
        if self.moe_impl == "grouped":
            self._grouped_routed = TEGroupedRoutedExperts(
                self.routed,
                dim=dim,
                expert_hidden=expert_hidden,
                num_experts=num_experts,
                dropout=dropout,
                align=16,
            )
        elif self.moe_impl == "cutlass":
            self._grouped_routed = CUTLASSGroupedRoutedExperts(
                self.routed,
                dim=dim,
                expert_hidden=expert_hidden,
                num_experts=num_experts,
                dropout=dropout,
            )
        elif self.moe_impl == "torch_grouped":
            self._grouped_routed = TorchGroupedRoutedExperts(
                self.routed,
                dim=dim,
                expert_hidden=expert_hidden,
                num_experts=num_experts,
                dropout=dropout,
            )
        elif self.moe_impl == "torchao_fp8":
            self._grouped_routed = TorchAOFP8GroupedRoutedExperts(
                self.routed,
                dim=dim,
                expert_hidden=expert_hidden,
                num_experts=num_experts,
                dropout=dropout,
            )
        elif self.moe_impl == "liger":
            self._grouped_routed = LigerGroupedRoutedExperts(
                self.routed,
                dim=dim,
                expert_hidden=expert_hidden,
                num_experts=num_experts,
                dropout=dropout,
            )
        self.last_aux_loss: torch.Tensor | None = None
        self.last_z_loss: torch.Tensor | None = None
        self.last_load: torch.Tensor | None = None
        # Top-1 expert route for low-frequency pathway/grokking diagnostics.
        # This is detached and bounded by the current microbatch size.
        self.last_top1_route: torch.Tensor | None = None

    def _run_expert(self, expert: nn.Module, tokens: torch.Tensor) -> torch.Tensor:
        if self.linear_backend != "transformer_engine":
            return expert(tokens)
        rows = int(tokens.shape[0])
        pad_rows = (-rows) % 16
        if pad_rows == 0:
            return expert(tokens)
        padded = torch.cat(
            [tokens, tokens.new_zeros((pad_rows, tokens.shape[-1]))],
            dim=0,
        )
        return expert(padded)[:rows]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        flat = x.reshape(-1, self.dim)
        router_logits = self.router(flat).float()
        affinity = torch.sigmoid(router_logits) if self.router_score == "sigmoid" else F.softmax(router_logits, dim=-1)
        selection_scores = affinity + self.routing_bias if self.balance_strategy in {"bias", "hybrid"} else affinity
        _, top_idx = selection_scores.topk(self.top_k, dim=-1)
        self.last_top1_route = top_idx[:, 0].detach()
        top_weight = affinity.gather(-1, top_idx)
        top_weight = top_weight / top_weight.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        if self.moe_impl in {
            "grouped",
            "cutlass",
            "torch_grouped",
            "torchao_fp8",
            "liger",
        }:
            if self._grouped_routed is None:
                raise RuntimeError("Grouped MoE bridge was not initialized")
            routed_out = self._grouped_routed(flat, top_idx, top_weight)
        else:
            routed_out = torch.zeros_like(flat)
            # Reference dispatch: each expert receives only the tokens routed to it.
            for expert_idx, expert in enumerate(self.routed):
                token_idx, slot_idx = torch.where(top_idx == expert_idx)
                if token_idx.numel() == 0:
                    continue
                expert_out = self._run_expert(expert, flat.index_select(0, token_idx))
                weight = top_weight[token_idx, slot_idx].to(expert_out.dtype).unsqueeze(-1)
                routed_out.index_add_(0, token_idx, expert_out * weight)

        shared_out = torch.zeros_like(flat)
        for expert in self.shared:
            shared_out = shared_out + self._run_expert(expert, flat)

        # Switch-style balancing signal plus router z-loss. The trainer decides the
        # coefficients, so these remain inspectable independently.
        load = fixed_bincount(top_idx, self.num_experts, dtype=torch.float32)
        load = load / float(top_idx.numel())
        importance = affinity / affinity.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        importance = importance.mean(dim=0)
        self.last_aux_loss = self.num_experts * torch.sum(importance * load.detach())
        self.last_z_loss = torch.mean(torch.logsumexp(router_logits, dim=-1).square())
        self.last_load = load.detach()
        if self.training and self.balance_strategy in {"bias", "hybrid"}:
            self.load_accumulator.add_(load.detach())
            self.load_batches.add_(1.0)
        return (routed_out + shared_out).reshape(original_shape)

    @torch.no_grad()
    def update_routing_bias(self) -> torch.Tensor | None:
        if self.balance_strategy not in {"bias", "hybrid"}:
            return None
        # This is called after at least one training forward. Avoid Tensor.item(),
        # which forced a GPU-to-CPU synchronization at every optimizer step.
        mean_load = self.load_accumulator / self.load_batches.clamp_min(1.0)
        target = torch.full_like(mean_load, 1.0 / self.num_experts)
        # Overloaded experts receive a lower selection-only bias; underloaded experts
        # receive a higher one. Gating weights still come from the unbiased affinity.
        self.routing_bias.add_(torch.sign(target - mean_load), alpha=self.bias_update_speed)
        self.routing_bias.sub_(self.routing_bias.mean())
        self.load_accumulator.zero_()
        self.load_batches.zero_()
        return mean_load

    def active_parameter_count(self) -> int:
        from asterlm.quantization.loqt import effective_parameter_count

        router = effective_parameter_count(self.router)
        shared = sum(effective_parameter_count(expert) for expert in self.shared)
        routed_one = effective_parameter_count(self.routed[0])
        return router + shared + self.top_k * routed_one
