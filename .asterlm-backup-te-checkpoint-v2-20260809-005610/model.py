from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .cache import AsterCache
from .config import AsterConfig
from .layers.block_attnres import DepthResidualMixer
from .layers.ffn import SwiGLU
from .layers.kda import KDA
from .layers.latent_attention import LatentAttention
from .layers.mtp import MultiTokenPredictor
from .layers.moe import DeepSeekStyleMoE
from .layers.norm import build_norm


@dataclass
class AsterOutput:
    logits: torch.Tensor | None
    loss: torch.Tensor | None = None
    main_loss: torch.Tensor | None = None
    mtp_loss: torch.Tensor | None = None
    mtp_logits: list[torch.Tensor] | None = None
    router_aux_loss: torch.Tensor | None = None
    router_z_loss: torch.Tensor | None = None
    expert_load: torch.Tensor | None = None
    cache: AsterCache | None = None
    hidden_states: torch.Tensor | None = None




def _masked_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int) -> torch.Tensor:
    """Mean CE over valid labels; returns a differentiable zero when every label is ignored."""
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)
    losses = F.cross_entropy(flat_logits, flat_labels, ignore_index=ignore_index, reduction="none")
    valid = flat_labels.ne(ignore_index)
    return losses.sum() / valid.sum().clamp_min(1)

class AsterBlock(nn.Module):
    def __init__(self, config: AsterConfig, kind: str, layer_idx: int, kda_idx: int | None) -> None:
        super().__init__()
        self.kind = kind
        self.layer_idx = layer_idx
        self.norm_mixer = build_norm(config.d_model, config.rms_eps, config.norm_type)
        self.norm_ffn = build_norm(config.d_model, config.rms_eps, config.norm_type)
        if kind == "kda":
            if kda_idx is None:
                raise ValueError("kda_idx is required for KDA blocks")
            self.mixer: nn.Module = KDA(config, kda_idx)
        elif kind == "latent":
            self.mixer = LatentAttention(config, layer_idx)
        else:
            raise ValueError(f"Unknown block kind: {kind}")
        use_moe = (
            config.ffn_type == "moe"
            and layer_idx >= config.moe_first_dense_layers
            and (layer_idx - config.moe_first_dense_layers) % config.moe_every == 0
        )
        self.ffn = (
            DeepSeekStyleMoE(
                config.d_model,
                config.moe_expert_hidden,
                config.moe_num_experts,
                config.moe_top_k,
                config.moe_shared_experts,
                config.ffn_dropout,
                config.moe_router_score,
                config.moe_balance_strategy,
                config.moe_router_bias_update_speed,
                config.ffn_backend,
                config.loqt_rank,
                config.loqt_alpha,
                config.loqt_group_size,
                config.init_std,
            )
            if use_moe
            else SwiGLU(
                config.d_model,
                config.ffn_hidden,
                config.ffn_dropout,
                config.ffn_backend,
                loqt_rank=config.loqt_rank,
                loqt_alpha=config.loqt_alpha,
                loqt_group_size=config.loqt_group_size,
                init_std=config.init_std,
            )
        )
        self.residual_dropout = nn.Dropout(config.residual_dropout)

    def forward(
        self,
        hidden: torch.Tensor,
        position_ids: torch.Tensor,
        cache: AsterCache | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        normed = self.norm_mixer(hidden)
        if self.kind == "kda":
            mixed = self.mixer(normed, cache=cache, use_cache=use_cache)
        else:
            layer_cache = None if cache is None else cache.latent_layer(self.layer_idx)
            mixed = self.mixer(normed, position_ids, cache=layer_cache, use_cache=use_cache)
        hidden = hidden + self.residual_dropout(mixed)
        hidden = hidden + self.residual_dropout(self.ffn(self.norm_ffn(hidden)))
        return hidden


class AsterLM(nn.Module):
    def __init__(self, config: AsterConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.embedding_in_proj = (
            nn.Linear(config.d_model, config.d_model, bias=False)
            if config.embedding_projection
            else nn.Identity()
        )
        self.embedding_out_proj = (
            nn.Linear(config.d_model, config.d_model, bias=False)
            if config.embedding_projection
            else nn.Identity()
        )
        self.embedding_dropout = nn.Dropout(config.residual_dropout)

        blocks: list[AsterBlock] = []
        kda_idx = 0
        for layer_idx, kind in enumerate(config.pattern):
            blocks.append(AsterBlock(config, kind, layer_idx, kda_idx if kind == "kda" else None))
            if kind == "kda":
                kda_idx += 1
        self.blocks = nn.ModuleList(blocks)
        self.n_kda_layers = kda_idx

        self.use_block_attnres = config.use_block_attnres
        self.attnres_block_size = config.attnres_block_size
        if self.use_block_attnres:
            n_groups = math.ceil(config.n_layers / config.attnres_block_size)
            self.depth_mixers = nn.ModuleList(
                [
                    DepthResidualMixer(
                        config.d_model,
                        config.attnres_key_dim,
                        max_states=group_idx + 1,
                        eps=config.rms_eps,
                        norm_type=config.norm_type,
                    )
                    for group_idx in range(1, n_groups)
                ]
            )
        else:
            self.depth_mixers = nn.ModuleList()

        self.final_norm = build_norm(config.d_model, config.rms_eps, config.norm_type)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        self.mtp = (
            MultiTokenPredictor(
                config.d_model,
                config.mtp_rank,
                config.mtp_depth,
                config.rms_eps,
                config.linear_backend,
                config.norm_type,
            )
            if config.mtp_depth > 0
            else None
        )
        self.apply(self._initialize_module)
        self._initialize_embedding_projections()
        self._scale_residual_projections()


    def _initialize_embedding_projections(self) -> None:
        """OSP uses orthogonally initialized input/output embedding rotations."""
        if not self.config.embedding_projection:
            return
        assert isinstance(self.embedding_in_proj, nn.Linear)
        assert isinstance(self.embedding_out_proj, nn.Linear)
        nn.init.orthogonal_(self.embedding_in_proj.weight)
        nn.init.orthogonal_(self.embedding_out_proj.weight)

    def _initialize_module(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear) or getattr(module, "_aster_linear", False):
            if hasattr(module, "weight") and module.weight is not None:
                nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)

    def _scale_residual_projections(self) -> None:
        scale = self.config.residual_init_scale
        if scale is None:
            scale = 1.0 / math.sqrt(2 * self.config.n_layers)
        with torch.no_grad():
            for module in self.modules():
                if not getattr(module, "_is_residual_projection", False):
                    continue
                if hasattr(module, "scale_effective_weight"):
                    module.scale_effective_weight(scale)
                elif hasattr(module, "weight"):
                    module.weight.mul_(scale)

    @property
    def uses_fla(self) -> bool:
        return any(isinstance(block.mixer, KDA) and block.mixer.uses_fla for block in self.blocks)

    def make_cache(self) -> AsterCache:
        return AsterCache.create(use_fla=self.uses_fla, config=self.config)

    def parameter_count(self, trainable_only: bool = True) -> int:
        params = self.parameters()
        if trainable_only:
            params = (p for p in params if p.requires_grad)
        return sum(p.numel() for p in params)

    def effective_parameter_count(self) -> int:
        """Count full logical matrices even when LoQT stores them packed in buffers."""
        from .quantization.loqt import effective_parameter_count

        return effective_parameter_count(self)

    @torch.no_grad()
    def folded_embedding_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return input embedding and output-head weights with OSP rotations absorbed.

        The returned matrices can be installed in an otherwise shape-identical model
        configured with ``embedding_projection=False`` and ``tie_embeddings=False``.
        """
        embedding = self.token_embedding.weight.detach().float()
        head = self.lm_head.weight.detach().float()
        if isinstance(self.embedding_in_proj, nn.Linear):
            embedding = embedding @ self.embedding_in_proj.weight.detach().float().T
        if isinstance(self.embedding_out_proj, nn.Linear):
            head = head @ self.embedding_out_proj.weight.detach().float()
        return embedding, head

    def _run_block(
        self,
        block: AsterBlock,
        hidden: torch.Tensor,
        position_ids: torch.Tensor,
        cache: AsterCache | None,
        use_cache: bool,
    ) -> torch.Tensor:
        if self.config.gradient_checkpointing and self.training and not use_cache:
            def custom_forward(h: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
                return block(h, p, cache=None, use_cache=False)

            return checkpoint(custom_forward, hidden, position_ids, use_reentrant=False)
        return block(hidden, position_ids, cache=cache, use_cache=use_cache)

    def _projected_cross_entropy(
        self,
        hidden: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: int,
    ) -> torch.Tensor:
        """Chunk the vocabulary projection and checkpoint each CE chunk.

        A [B,T,V] tensor is often the largest non-parameter allocation in compact-LM
        training. Chunking limits peak logits memory; checkpointing avoids retaining
        each chunk's softmax activations until backward.
        """
        chunk_size = self.config.lm_loss_chunk_size
        losses: list[torch.Tensor] = []
        valid = labels.ne(ignore_index).sum().clamp_min(1)

        def chunk_loss(h: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            logits = self.lm_head(h)
            return F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                target.reshape(-1),
                ignore_index=ignore_index,
                reduction="sum",
            )

        for start in range(0, hidden.shape[1], chunk_size):
            stop = min(start + chunk_size, hidden.shape[1])
            h = hidden[:, start:stop]
            target = labels[:, start:stop]
            if self.config.gradient_checkpointing and self.training:
                loss_sum = checkpoint(chunk_loss, h, target, use_reentrant=False)
            else:
                loss_sum = chunk_loss(h, target)
            losses.append(loss_sum)
        return torch.stack(losses).sum() / valid

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        cache: AsterCache | None = None,
        use_cache: bool = False,
        return_mtp: bool = False,
        return_logits: bool = True,
        return_hidden: bool = False,
        ignore_index: int = -100,
    ) -> AsterOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        bsz, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len and cache is None:
            raise ValueError(
                f"Sequence length {seq_len} exceeds configured maximum {self.config.max_seq_len}. "
                "Use RoPE scaling and raise max_seq_len deliberately."
            )
        if use_cache and cache is None:
            cache = self.make_cache()
        start = 0 if cache is None else cache.seen_tokens
        position_ids = torch.arange(start, start + seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, -1)

        hidden = self.embedding_dropout(self.embedding_in_proj(self.token_embedding(input_ids)))
        depth_states: list[torch.Tensor] = [hidden]
        depth_mixer_idx = 0
        for layer_idx, block in enumerate(self.blocks):
            if (
                self.use_block_attnres
                and layer_idx > 0
                and layer_idx % self.attnres_block_size == 0
            ):
                depth_states.append(hidden)
                hidden = self.depth_mixers[depth_mixer_idx](depth_states)
                depth_mixer_idx += 1
            hidden = self._run_block(block, hidden, position_ids, cache, use_cache)

        hidden = self.final_norm(hidden)
        # Callers such as RLVR can request normalized hidden states and compute only
        # selected-token log-probabilities in bounded vocabulary chunks.  Defaults
        # remain unchanged for ordinary inference, where return_logits=True.
        need_logits = return_logits or return_mtp
        need_projected_hidden = need_logits or labels is not None
        projected_hidden = self.embedding_out_proj(hidden) if need_projected_hidden else None
        logits = self.lm_head(projected_hidden) if need_logits and projected_hidden is not None else None

        main_loss = None
        mtp_loss = None
        total_loss = None
        moe_modules = [block.ffn for block in self.blocks if isinstance(block.ffn, DeepSeekStyleMoE)]
        router_aux_loss = None
        router_z_loss = None
        expert_load = None
        if moe_modules:
            aux_terms = [m.last_aux_loss for m in moe_modules if m.last_aux_loss is not None]
            z_terms = [m.last_z_loss for m in moe_modules if m.last_z_loss is not None]
            loads = [m.last_load for m in moe_modules if m.last_load is not None]
            if aux_terms:
                router_aux_loss = torch.stack(aux_terms).mean()
            if z_terms:
                router_z_loss = torch.stack(z_terms).mean()
            if loads:
                expert_load = torch.stack(loads).mean(dim=0)
        mtp_logits: list[torch.Tensor] | None = [] if return_mtp else None
        if labels is not None:
            if logits is not None and return_logits:
                main_loss = _masked_cross_entropy(logits, labels, ignore_index)
            else:
                if projected_hidden is None:
                    raise RuntimeError("projected hidden states are required for language-model loss")
                main_loss = self._projected_cross_entropy(projected_hidden, labels, ignore_index)
            total_loss = main_loss
            if router_aux_loss is not None and self.config.moe_balance_strategy in {"aux_loss", "hybrid"}:
                total_loss = total_loss + self.config.moe_aux_loss_weight * router_aux_loss
            if router_z_loss is not None:
                total_loss = total_loss + self.config.moe_router_z_loss_weight * router_z_loss

        if self.mtp is not None and (labels is not None or return_mtp):
            losses: list[torch.Tensor] = []
            future = hidden
            for head_idx, head in enumerate(self.mtp.heads):
                if self.config.gradient_checkpointing and self.training:
                    future = checkpoint(head, future, use_reentrant=False)
                else:
                    future = head(future)
                if return_mtp and mtp_logits is not None:
                    mtp_logits.append(self.lm_head(self.embedding_out_proj(future)))
                if labels is not None:
                    shift = head_idx + 1
                    if labels.shape[1] > shift:
                        losses.append(
                            self._projected_cross_entropy(
                                self.embedding_out_proj(future[:, :-shift]), labels[:, shift:], ignore_index
                            )
                        )
            if losses:
                mtp_loss = torch.stack(losses).mean()
                total_loss = total_loss + self.config.mtp_loss_weight * mtp_loss

        if use_cache and cache is not None:
            cache.seen_tokens += seq_len
        return AsterOutput(
            logits=logits,
            loss=total_loss,
            main_loss=main_loss,
            mtp_loss=mtp_loss,
            mtp_logits=mtp_logits,
            router_aux_loss=router_aux_loss,
            router_z_loss=router_z_loss,
            expert_load=expert_load,
            cache=cache,
            hidden_states=hidden if return_hidden else None,
        )


    @torch.no_grad()
    def moe_pathway_stats(self, sample_tokens: int = 2048) -> dict[str, float]:
        """Summarize token-to-expert pathways across MoE layers.

        These are inexpensive monitoring signals inspired by recent practical-LLM
        grokking work. They are not presented as an exact reproduction of any paper's
        metric definitions; their purpose is to reveal route collapse, instability,
        and delayed emergence of reusable cross-layer pathways.
        """
        routes = [
            block.ffn.last_top1_route
            for block in self.blocks
            if isinstance(block.ffn, DeepSeekStyleMoE)
            and block.ffn.last_top1_route is not None
        ]
        if len(routes) < 2:
            return {}
        count = min(route.numel() for route in routes)
        if count <= 0:
            return {}
        count = min(count, sample_tokens)
        # Evenly sample positions so a long packed sequence is represented end to end.
        source_count = min(route.numel() for route in routes)
        indices = torch.linspace(
            0,
            source_count - 1,
            steps=count,
            device=routes[0].device,
        ).long()
        pathway = torch.stack([route.index_select(0, indices) for route in routes], dim=1)
        adjacent = pathway[:, 1:].eq(pathway[:, :-1]).float().mean()

        pair_count = pathway.shape[0] // 2
        pair_similarity = torch.tensor(0.0, device=pathway.device)
        if pair_count:
            pair_similarity = pathway[:pair_count].eq(pathway[-pair_count:]).float().mean()

        unique_fraction = pathway.shape[0] and (
            torch.unique(pathway, dim=0).shape[0] / pathway.shape[0]
        )
        entropies: list[torch.Tensor] = []
        for layer in range(pathway.shape[1]):
            counts = torch.bincount(
                pathway[:, layer], minlength=self.config.moe_num_experts
            ).float()
            probabilities = counts / counts.sum().clamp_min(1.0)
            entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
            entropies.append(entropy / math.log(self.config.moe_num_experts))

        return {
            "moe_path_adjacent_consistency": float(adjacent),
            "moe_path_pair_similarity": float(pair_similarity),
            "moe_path_unique_fraction": float(unique_fraction),
            "moe_path_layer_entropy_normalized": float(torch.stack(entropies).mean()),
            "moe_path_tokens_sampled": float(pathway.shape[0]),
            "moe_path_layers": float(pathway.shape[1]),
        }


    @torch.no_grad()
    def update_moe_router_biases(self) -> dict[str, float]:
        loads = []
        biases = []
        for block in self.blocks:
            if isinstance(block.ffn, DeepSeekStyleMoE):
                load = block.ffn.update_routing_bias()
                if load is not None:
                    loads.append(load)
                    biases.append(block.ffn.routing_bias)
        if not loads:
            return {}
        load = torch.stack(loads).mean(dim=0)
        bias = torch.stack(biases).mean(dim=0)
        uniform = 1.0 / load.numel()
        return {
            "moe_load_max_ratio": float(load.max() / uniform),
            "moe_load_min_ratio": float(load.min() / uniform),
            "moe_load_cv": float(load.std(unbiased=False) / load.mean().clamp_min(1e-9)),
            "moe_routing_bias_absmax": float(bias.abs().max()),
        }

    @torch.no_grad()
    def apply_qk_clip(self, tau: float | None = None) -> dict[str, float]:
        tau = self.config.qk_clip_tau if tau is None else tau
        total_heads = 0.0
        maximum = 0.0
        for block in self.blocks:
            if isinstance(block.mixer, LatentAttention):
                stats = block.mixer.apply_qk_clip(tau)
                total_heads += stats["heads_clipped"]
                maximum = max(maximum, stats["max_logit_before"])
        return {"qk_heads_clipped": total_heads, "qk_max_logit": maximum}

    def active_parameter_count(self) -> int:
        """Logical parameters used for one token, accounting for sparse MoE routing."""
        from .quantization.loqt import effective_parameter_count

        total = effective_parameter_count(self)
        for block in self.blocks:
            if isinstance(block.ffn, DeepSeekStyleMoE):
                total -= effective_parameter_count(block.ffn)
                total += block.ffn.active_parameter_count()
        return total

    def architecture_summary(self) -> dict[str, Any]:
        pattern = self.config.pattern
        return {
            "trainable_parameters": self.parameter_count(),
            "effective_parameters": self.effective_parameter_count(),
            "layers": len(pattern),
            "kda_layers": pattern.count("kda"),
            "latent_attention_layers": pattern.count("latent"),
            "mtp_depth": self.config.mtp_depth,
            "fla_enabled": self.uses_fla,
            "max_sequence_length": self.config.max_seq_len,
            "ffn_type": self.config.ffn_type,
            "moe_layers": sum(isinstance(block.ffn, DeepSeekStyleMoE) for block in self.blocks),
            "active_parameters_estimate": self.active_parameter_count(),
            "attention_window": self.config.attention_window,
            "latent_cache_width": self.config.latent_rank + self.config.rope_dim,
        }
