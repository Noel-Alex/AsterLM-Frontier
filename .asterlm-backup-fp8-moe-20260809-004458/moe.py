from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .ffn import SwiGLU
from .linear import build_linear


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
        self.router = build_linear(dim, num_experts, bias=False, backend=linear_backend)
        self.register_buffer("routing_bias", torch.zeros(num_experts, dtype=torch.float32))
        self.register_buffer("load_accumulator", torch.zeros(num_experts, dtype=torch.float32), persistent=False)
        self.register_buffer("load_batches", torch.zeros((), dtype=torch.float32), persistent=False)
        ffn_kwargs = dict(
            loqt_rank=loqt_rank,
            loqt_alpha=loqt_alpha,
            loqt_group_size=loqt_group_size,
            init_std=init_std,
        )
        self.routed = nn.ModuleList(
            [SwiGLU(dim, expert_hidden, dropout, linear_backend, **ffn_kwargs) for _ in range(num_experts)]
        )
        self.shared = nn.ModuleList(
            [SwiGLU(dim, expert_hidden, dropout, linear_backend, **ffn_kwargs) for _ in range(shared_experts)]
        )
        self.last_aux_loss: torch.Tensor | None = None
        self.last_z_loss: torch.Tensor | None = None
        self.last_load: torch.Tensor | None = None
        # Top-1 expert route for low-frequency pathway/grokking diagnostics.
        # This is detached and bounded by the current microbatch size.
        self.last_top1_route: torch.Tensor | None = None

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

        routed_out = torch.zeros_like(flat)
        # Reference dispatch: each expert receives only the tokens routed to it.
        for expert_idx, expert in enumerate(self.routed):
            token_idx, slot_idx = torch.where(top_idx == expert_idx)
            if token_idx.numel() == 0:
                continue
            expert_out = expert(flat.index_select(0, token_idx))
            weight = top_weight[token_idx, slot_idx].to(expert_out.dtype).unsqueeze(-1)
            routed_out.index_add_(0, token_idx, expert_out * weight)

        shared_out = torch.zeros_like(flat)
        for expert in self.shared:
            shared_out = shared_out + expert(flat)

        # Switch-style balancing signal plus router z-loss. The trainer decides the
        # coefficients, so these remain inspectable independently.
        dispatch = F.one_hot(top_idx, num_classes=self.num_experts).float().sum(dim=1) / self.top_k
        load = dispatch.mean(dim=0)
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
        if self.balance_strategy not in {"bias", "hybrid"} or self.load_batches.item() == 0:
            return None
        mean_load = self.load_accumulator / self.load_batches
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
