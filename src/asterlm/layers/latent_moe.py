from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from asterlm.quantization.loqt import effective_parameter_count

from .ffn import SiTUGLU, SwiGLU
from .linear import build_linear, mark_residual
from .moe_grouped_cutlass import CUTLASSGroupedRoutedExperts
from .moe_grouped_liger import LigerGroupedRoutedExperts
from .moe_grouped_te import TEGroupedRoutedExperts
from .moe_grouped_torch import TorchGroupedRoutedExperts
from .norm import RMSNorm
from .routing import fixed_bincount


class LatentMoE(nn.Module):
    """NVIDIA-style LatentMoE adapted to Aster's single-GPU execution model.

    Routed tokens are compressed from the model hidden size ``d`` to a latent size
    ``ell`` before expert computation. Routing itself and always-on shared experts stay
    in the original hidden dimension. This follows the core LatentMoE equation from
    Elango et al. (2026):

        W_up * sum_i p_i E_i(W_down x; ell) + sum_j E_shared_j(x; d)

    The router is deliberately computed from the original token x. Kimi K3's complete
    Stable LatentMoE recipe is represented by three independent switches: routed
    post-normalization, SiTU-GLU experts, and Quantile Balancing. Keeping them
    independent permits causal ablations instead of relabeling post-normalization alone.
    """

    def __init__(
        self,
        dim: int,
        latent_dim: int,
        expert_hidden: int,
        num_experts: int,
        top_k: int,
        shared_experts: int = 1,
        dropout: float = 0.0,
        router_score: str = "sigmoid",
        balance_strategy: str = "bias",
        bias_update_speed: float = 0.001,
        linear_backend: str = "torch",
        moe_impl: str = "reference",
        loqt_rank: int = 32,
        loqt_alpha: float = 32.0,
        loqt_group_size: int = 64,
        init_std: float = 0.02,
        norm_eps: float = 1e-6,
        post_norm: bool = False,
        activation: str = "swiglu",
        situ_beta_gate: float = 4.0,
        situ_beta_up: float = 25.0,
        quantile_bins: int = 256,
        quantile_margin_bound: float = 4.0,
    ) -> None:
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError("LatentMoE requires 1 <= top_k <= num_experts")
        if not 0 < latent_dim < dim:
            raise ValueError("LatentMoE latent_dim must be positive and smaller than dim")
        if shared_experts < 0:
            raise ValueError("shared_experts must be non-negative")
        if moe_impl not in {
            "reference",
            "grouped",
            "cutlass",
            "torch_grouped",
            "liger",
        }:
            raise ValueError(
                "moe_impl must be reference, grouped, cutlass, torch_grouped, or liger"
            )
        if moe_impl == "grouped" and linear_backend != "transformer_engine":
            raise ValueError("grouped LatentMoE requires Transformer Engine expert linears")
        if activation not in {"swiglu", "situ_glu"}:
            raise ValueError("activation must be swiglu or situ_glu")
        if moe_impl == "grouped" and activation != "swiglu":
            raise ValueError(
                "Transformer Engine's fused grouped path supports SwiGLU only; "
                "use the CUTLASS grouped path for SiTU-GLU"
            )
        if moe_impl == "liger" and activation != "swiglu":
            raise ValueError(
                "LigerExperts 0.8 supports SwiGLU only; use CUTLASS for SiTU-GLU"
            )
        if quantile_bins < 16 or quantile_margin_bound <= 0:
            raise ValueError("invalid Quantile Balancing histogram settings")

        self.dim = int(dim)
        self.latent_dim = int(latent_dim)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.router_score = router_score
        self.balance_strategy = balance_strategy
        self.bias_update_speed = float(bias_update_speed)
        self.linear_backend = linear_backend
        self.moe_impl = moe_impl
        self.activation = activation
        self.quantile_bins = int(quantile_bins)
        self.quantile_margin_bound = float(quantile_margin_bound)

        router_backend = "torch" if linear_backend == "transformer_engine" else linear_backend
        self.router = build_linear(dim, num_experts, bias=False, backend=router_backend)
        self.down_proj = build_linear(dim, latent_dim, bias=False, backend=linear_backend)
        # Baseline LatentMoE leaves the routed aggregate untouched. Kimi K3's
        # Stable LatentMoE normalizes that aggregate before the shared up projection;
        # keep this explicit and ablatable rather than silently changing the baseline.
        self.routed_post_norm = RMSNorm(latent_dim, norm_eps) if post_norm else nn.Identity()
        self.up_proj = mark_residual(
            build_linear(latent_dim, dim, bias=False, backend=linear_backend)
        )

        self.register_buffer("routing_bias", torch.zeros(num_experts, dtype=torch.float32))
        self.register_buffer(
            "load_accumulator", torch.zeros(num_experts, dtype=torch.float32), persistent=False
        )
        self.register_buffer("load_batches", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer(
            "quantile_histogram",
            torch.zeros(num_experts, quantile_bins, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "quantile_tokens", torch.zeros((), dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "quantile_clipped", torch.zeros((), dtype=torch.float32), persistent=False
        )

        ffn_kwargs = {
            "loqt_rank": loqt_rank,
            "loqt_alpha": loqt_alpha,
            "loqt_group_size": loqt_group_size,
            "init_std": init_std,
        }
        expert_type = SiTUGLU if activation == "situ_glu" else SwiGLU
        if activation == "situ_glu":
            ffn_kwargs.update(beta_gate=situ_beta_gate, beta_up=situ_beta_up)
        self.routed = nn.ModuleList(
            [
                expert_type(latent_dim, expert_hidden, dropout, linear_backend, **ffn_kwargs)
                for _ in range(num_experts)
            ]
        )
        self.shared = nn.ModuleList(
            [
                expert_type(dim, expert_hidden, dropout, linear_backend, **ffn_kwargs)
                for _ in range(shared_experts)
            ]
        )
        self._grouped_routed = None
        if self.moe_impl == "grouped":
            self._grouped_routed = TEGroupedRoutedExperts(
                self.routed,
                dim=latent_dim,
                expert_hidden=expert_hidden,
                num_experts=num_experts,
                dropout=dropout,
                align=16,
            )
        elif self.moe_impl == "cutlass":
            self._grouped_routed = CUTLASSGroupedRoutedExperts(
                self.routed,
                dim=latent_dim,
                expert_hidden=expert_hidden,
                num_experts=num_experts,
                dropout=dropout,
            )
        elif self.moe_impl == "torch_grouped":
            self._grouped_routed = TorchGroupedRoutedExperts(
                self.routed,
                dim=latent_dim,
                expert_hidden=expert_hidden,
                num_experts=num_experts,
                dropout=dropout,
            )
        elif self.moe_impl == "liger":
            self._grouped_routed = LigerGroupedRoutedExperts(
                self.routed,
                dim=latent_dim,
                expert_hidden=expert_hidden,
                num_experts=num_experts,
                dropout=dropout,
            )

        self.last_aux_loss: torch.Tensor | None = None
        self.last_z_loss: torch.Tensor | None = None
        self.last_load: torch.Tensor | None = None
        self.last_top1_route: torch.Tensor | None = None
        self.last_quantile_clipped_fraction: torch.Tensor | None = None

    @torch.no_grad()
    def _accumulate_quantile_histogram(
        self, affinity: torch.Tensor, cutoff: torch.Tensor
    ) -> None:
        margins = affinity - cutoff.unsqueeze(-1)
        bound = self.quantile_margin_bound
        clipped = margins.clamp(-bound, bound)
        self.quantile_clipped.add_((clipped != margins).sum())
        scaled = ((clipped + bound) * (self.quantile_bins / (2.0 * bound))).floor()
        bins = scaled.to(torch.long).clamp_(0, self.quantile_bins - 1)
        offsets = (
            torch.arange(self.num_experts, device=bins.device, dtype=torch.long)
            * self.quantile_bins
        )
        flat_bins = (bins + offsets).reshape(-1)
        histogram = fixed_bincount(
            flat_bins,
            self.num_experts * self.quantile_bins,
            dtype=self.quantile_histogram.dtype,
        ).reshape(self.num_experts, self.quantile_bins)
        self.quantile_histogram.add_(histogram)
        self.quantile_tokens.add_(float(affinity.shape[0]))

    def _run_expert(self, expert: nn.Module, tokens: torch.Tensor) -> torch.Tensor:
        if self.linear_backend != "transformer_engine":
            return expert(tokens)
        rows = int(tokens.shape[0])
        pad_rows = (-rows) % 16
        if pad_rows == 0:
            return expert(tokens)
        padded = torch.cat(
            (tokens, tokens.new_zeros((pad_rows, tokens.shape[-1]))),
            dim=0,
        )
        return expert(padded)[:rows]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        flat = x.reshape(-1, self.dim)

        # The LatentMoE paper computes routing logits from full-width x.
        router_logits = self.router(flat).float()
        affinity = (
            torch.sigmoid(router_logits)
            if self.router_score == "sigmoid"
            else F.softmax(router_logits, dim=-1)
        )
        uses_selection_bias = self.balance_strategy in {"bias", "hybrid", "quantile"}
        selection_scores = (
            affinity + self.routing_bias
            if uses_selection_bias
            else affinity
        )
        selection_width = (
            self.top_k + 1
            if self.balance_strategy == "quantile" and self.top_k < self.num_experts
            else self.top_k
        )
        top_values, top_indices = selection_scores.topk(selection_width, dim=-1)
        top_idx = top_indices[:, : self.top_k]
        if self.training and self.balance_strategy == "quantile" and selection_width > self.top_k:
            self._accumulate_quantile_histogram(affinity, top_values[:, self.top_k])
        self.last_top1_route = top_idx[:, 0].detach()
        top_weight = affinity.gather(-1, top_idx)
        top_weight = top_weight / top_weight.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        latent = self.down_proj(flat)
        if self.moe_impl in {"grouped", "cutlass", "torch_grouped", "liger"}:
            if self._grouped_routed is None:
                raise RuntimeError("Grouped LatentMoE bridge was not initialized")
            routed_latent = self._grouped_routed(latent, top_idx, top_weight)
        else:
            routed_latent = torch.zeros_like(latent)
            for expert_idx, expert in enumerate(self.routed):
                token_idx, slot_idx = torch.where(top_idx == expert_idx)
                if token_idx.numel() == 0:
                    continue
                expert_out = self._run_expert(expert, latent.index_select(0, token_idx))
                weight = top_weight[token_idx, slot_idx].to(expert_out.dtype).unsqueeze(-1)
                routed_latent.index_add_(0, token_idx, expert_out * weight)

        routed_out = self.up_proj(self.routed_post_norm(routed_latent))

        shared_out = torch.zeros_like(flat)
        for expert in self.shared:
            shared_out = shared_out + self._run_expert(expert, flat)

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
        if self.balance_strategy == "quantile":
            if self.top_k >= self.num_experts:
                return None
            histogram = self.quantile_histogram.clone()
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(histogram)
            counts = histogram.sum(dim=-1)
            probability = 1.0 - (self.top_k / self.num_experts)
            ranks = (counts * probability).ceil().clamp_min(1.0).unsqueeze(-1)
            cumulative = histogram.cumsum(dim=-1)
            bin_index = (cumulative >= ranks).to(torch.int64).argmax(dim=-1)
            width = 2.0 * self.quantile_margin_bound / self.quantile_bins
            quantile = -self.quantile_margin_bound + (bin_index.float() + 0.5) * width
            self.routing_bias.copy_(-quantile)
            self.routing_bias.sub_(self.routing_bias.mean())
            denominator = (self.quantile_tokens * self.num_experts).clamp_min(1.0)
            self.last_quantile_clipped_fraction = self.quantile_clipped / denominator
            self.quantile_histogram.zero_()
            self.quantile_tokens.zero_()
            self.quantile_clipped.zero_()
            return counts / counts.sum().clamp_min(1.0)
        if self.balance_strategy not in {"bias", "hybrid"}:
            return None
        mean_load = self.load_accumulator / self.load_batches.clamp_min(1.0)
        target = torch.full_like(mean_load, 1.0 / self.num_experts)
        self.routing_bias.add_(torch.sign(target - mean_load), alpha=self.bias_update_speed)
        self.routing_bias.sub_(self.routing_bias.mean())
        self.load_accumulator.zero_()
        self.load_batches.zero_()
        return mean_load

    def active_parameter_count(self) -> int:
        router = effective_parameter_count(self.router)
        projections = effective_parameter_count(self.down_proj) + effective_parameter_count(self.up_proj)
        shared = sum(effective_parameter_count(expert) for expert in self.shared)
        routed_one = effective_parameter_count(self.routed[0])
        return router + projections + shared + self.top_k * routed_one
