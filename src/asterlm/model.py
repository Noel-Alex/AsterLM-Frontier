from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .cache import AsterCache
from .config import AsterConfig
from .layers.attnres_vnext import AttnResMix
from .layers.ffn import SwiGLU
from .layers.gdn2 import GDN2
from .layers.kda import KDA
from .layers.latent_attention import LatentAttention
from .layers.latent_moe import LatentMoE
from .layers.moe import DeepSeekStyleMoE
from .layers.mtp import MultiTokenPredictor
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

def _aster_activation_checkpoint(
    config: AsterConfig,
    function,
    *args: torch.Tensor,
    use_reentrant: bool = False,
):
    # Use TE-aware activation recomputation when Transformer Engine is active.
    uses_te = (
        config.linear_backend == "transformer_engine"
        or config.ffn_backend == "transformer_engine"
    )
    if uses_te:
        try:
            import transformer_engine.pytorch as te
        except ImportError as exc:
            raise ImportError(
                "Transformer Engine checkpointing requested but transformer_engine "
                "is not importable."
            ) from exc
        return te.distributed.checkpoint(
            function,
            *args,
            use_reentrant=use_reentrant,
        )
    return checkpoint(function, *args, use_reentrant=use_reentrant)


class AsterBlock(nn.Module):
    def __init__(self, config: AsterConfig, kind: str, layer_idx: int, kda_idx: int | None) -> None:
        super().__init__()
        self.kind = kind
        self.layer_idx = layer_idx
        self.norm_mixer = build_norm(config.d_model, config.rms_eps, config.norm_type)
        self.norm_ffn = build_norm(config.d_model, config.rms_eps, config.norm_type)
        if kind == "kda":
            if kda_idx is None:
                raise ValueError("recurrent index is required for KDA blocks")
            self.mixer: nn.Module = KDA(config, kda_idx)
        elif kind == "gdn2":
            if kda_idx is None:
                raise ValueError("recurrent index is required for GDN2 blocks")
            self.mixer = GDN2(config, kda_idx)
        elif kind == "latent":
            self.mixer = LatentAttention(config, layer_idx)
        else:
            raise ValueError(f"Unknown block kind: {kind}")
        use_moe = (
            config.ffn_type in {"moe", "latent_moe"}
            and layer_idx >= config.moe_first_dense_layers
            and (layer_idx - config.moe_first_dense_layers) % config.moe_every == 0
        )
        if use_moe and config.ffn_type == "latent_moe":
            self.ffn = LatentMoE(
                dim=config.d_model,
                latent_dim=config.latent_moe_dim or (config.d_model // 4),
                expert_hidden=config.moe_expert_hidden,
                num_experts=config.moe_num_experts,
                top_k=config.moe_top_k,
                shared_experts=config.moe_shared_experts,
                dropout=config.ffn_dropout,
                router_score=config.moe_router_score,
                balance_strategy=config.moe_balance_strategy,
                bias_update_speed=config.moe_router_bias_update_speed,
                linear_backend=config.ffn_backend,
                moe_impl=os.environ.get("ASTER_MOE_IMPL", "reference").strip().lower(),
                loqt_rank=config.loqt_rank,
                loqt_alpha=config.loqt_alpha,
                loqt_group_size=config.loqt_group_size,
                init_std=config.init_std,
                norm_eps=config.rms_eps,
                post_norm=config.latent_moe_post_norm,
                activation=config.moe_activation,
                situ_beta_gate=config.moe_situ_beta_gate,
                situ_beta_up=config.moe_situ_beta_up,
                quantile_bins=config.moe_quantile_bins,
                quantile_margin_bound=config.moe_quantile_margin_bound,
            )
        elif use_moe:
            self.ffn = DeepSeekStyleMoE(
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
        else:
            self.ffn = SwiGLU(
                config.d_model,
                config.ffn_hidden,
                config.ffn_dropout,
                config.ffn_backend,
                loqt_rank=config.loqt_rank,
                loqt_alpha=config.loqt_alpha,
                loqt_group_size=config.loqt_group_size,
                init_std=config.init_std,
            )
        self.residual_dropout = nn.Dropout(config.residual_dropout)
        self.use_attnres = config.use_block_attnres
        if self.use_attnres:
            self.attn_res_mix = AttnResMix(config.d_model, config.rms_eps)
            self.ffn_res_mix = AttnResMix(config.d_model, config.rms_eps)
        else:
            self.attn_res_mix = None
            self.ffn_res_mix = None

    def forward(
        self,
        hidden: torch.Tensor,
        position_ids: torch.Tensor,
        cache: AsterCache | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        normed = self.norm_mixer(hidden)
        if self.kind in {"kda", "gdn2"}:
            mixed = self.mixer(normed, cache=cache, use_cache=use_cache)
        else:
            layer_cache = None if cache is None else cache.latent_layer(self.layer_idx)
            mixed = self.mixer(normed, position_ids, cache=layer_cache, use_cache=use_cache)
        hidden = hidden + self.residual_dropout(mixed)
        hidden = hidden + self.residual_dropout(self.ffn(self.norm_ffn(hidden)))
        return hidden

    def forward_attnres(
        self,
        hidden: torch.Tensor,
        position_ids: torch.Tensor,
        depth_states: list[torch.Tensor] | None,
        cache: AsterCache | None = None,
        use_cache: bool = False,
        attnres_block_size: int = 4,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Faithful Block AttnRes over attention and FFN sublayers.

        ``hidden`` carries the running prefix sum. ``depth_states`` stores only
        completed block summaries, matching the current FLA/KDA reference.
        """
        if self.attn_res_mix is None or self.ffn_res_mix is None:
            raise RuntimeError("AttnRes sublayers were not initialized")

        prefix_sum: torch.Tensor | None = hidden
        states = None if depth_states is None else list(depth_states)

        # First attention sublayer is a single-source identity in AttnRes. Bypass
        # the mixer, but move the embedding into the completed-state list exactly
        # as the reference implementation does.
        if states is None:
            attn_input = self.norm_mixer(prefix_sum)
            states = [prefix_sum]
            prefix_sum = None
        else:
            residuals = [*states, prefix_sum]
            if (2 * self.layer_idx) % attnres_block_size == 0:
                states = residuals
                prefix_sum = None
            attn_input = self.norm_mixer(self.attn_res_mix(residuals))

        if self.kind in {"kda", "gdn2"}:
            mixed = self.mixer(attn_input, cache=cache, use_cache=use_cache)
        else:
            layer_cache = None if cache is None else cache.latent_layer(self.layer_idx)
            mixed = self.mixer(attn_input, position_ids, cache=layer_cache, use_cache=use_cache)
        mixed = self.residual_dropout(mixed)
        prefix_sum = mixed if prefix_sum is None else prefix_sum + mixed

        residuals = [*states, prefix_sum]
        if (2 * self.layer_idx + 1) % attnres_block_size == 0:
            states = residuals
            prefix_sum = None
        ffn_input = self.norm_ffn(self.ffn_res_mix(residuals))
        ffn_out = self.residual_dropout(self.ffn(ffn_input))
        prefix_sum = ffn_out if prefix_sum is None else prefix_sum + ffn_out
        return prefix_sum, states



class AsterLM(nn.Module):
    def __init__(
        self,
        config: AsterConfig,
        *,
        named_initialization_seed: int | None = None,
    ) -> None:
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
        recurrent_idx = 0
        self.n_kda_layers = 0
        self.n_gdn2_layers = 0
        for layer_idx, kind in enumerate(config.pattern):
            idx = recurrent_idx if kind in {"kda", "gdn2"} else None
            blocks.append(AsterBlock(config, kind, layer_idx, idx))
            if kind in {"kda", "gdn2"}:
                recurrent_idx += 1
            if kind == "kda":
                self.n_kda_layers += 1
            elif kind == "gdn2":
                self.n_gdn2_layers += 1
        self.blocks = nn.ModuleList(blocks)

        self.use_block_attnres = config.use_block_attnres
        self.attnres_block_size = config.attnres_block_size
        self.final_attnres = AttnResMix(config.d_model, config.rms_eps) if self.use_block_attnres else None

        self.final_norm = build_norm(config.d_model, config.rms_eps, config.norm_type)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        self.mtp = None
        self.mtp_hnorm = None
        self.mtp_enorm = None
        self.mtp_eh_proj = None
        self.mtp_deepseek_block = None
        self.mtp_final_norm = None
        if config.mtp_depth > 0:
            if config.mtp_architecture == "low_rank":
                self.mtp = MultiTokenPredictor(
                    config.d_model,
                    config.mtp_rank,
                    config.mtp_depth,
                    config.rms_eps,
                    config.linear_backend,
                    config.norm_type,
                )
            else:
                # DeepSeek-V3-style sequential MTP: normalize the main hidden state
                # and the *actual future token embedding*, concatenate, project 2d->d,
                # run one full model block, final-normalize, then reuse the LM head.
                self.mtp_hnorm = build_norm(config.d_model, config.rms_eps, config.norm_type)
                self.mtp_enorm = build_norm(config.d_model, config.rms_eps, config.norm_type)
                from .layers.linear import build_linear
                self.mtp_eh_proj = build_linear(
                    2 * config.d_model,
                    config.d_model,
                    bias=False,
                    backend=config.linear_backend,
                )
                recurrent_idx = self.n_kda_layers + self.n_gdn2_layers
                self.mtp_deepseek_block = AsterBlock(
                    config,
                    config.mtp_block_kind,
                    config.n_layers,
                    recurrent_idx if config.mtp_block_kind in {"kda", "gdn2"} else None,
                )
                self.mtp_final_norm = build_norm(config.d_model, config.rms_eps, config.norm_type)
        if named_initialization_seed is None:
            self.apply(self._initialize_module)
        else:
            self._initialize_modules_by_name(named_initialization_seed)
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

    @staticmethod
    def _qualified_initialization_seed(base_seed: int, module_name: str) -> int:
        digest = hashlib.blake2b(
            f"{base_seed}:{module_name}".encode(), digest_size=8
        ).digest()
        return int.from_bytes(digest, "little") % (2**63 - 1)

    def _initialize_modules_by_name(self, base_seed: int) -> None:
        """Initialize shared projections identically across architecture variants.

        KDA/FLA-specific time constants and other specialized tensors retain their
        upstream initialization. Only the same ordinary modules reset by Aster's
        historical ``apply`` pass are made name-deterministic.
        """

        for name, module in self.named_modules():
            is_linear = isinstance(module, nn.Linear) or getattr(
                module, "_aster_linear", False
            )
            if not is_linear and not isinstance(module, nn.Embedding):
                continue
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self._qualified_initialization_seed(base_seed, name))
                self._initialize_module(module)

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
        return any(
            (isinstance(block.mixer, KDA) and block.mixer.uses_fla)
            or isinstance(block.mixer, GDN2)
            for block in self.blocks
        )

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
        depth_states: list[torch.Tensor] | None = None,
    ):
        checkpointed = self.config.gradient_checkpointing and self.training and not use_cache
        if self.use_block_attnres:
            states = None if depth_states is None else list(depth_states)
            if checkpointed:
                def custom_forward(h: torch.Tensor, p: torch.Tensor, *state_tensors: torch.Tensor):
                    out, new_states = block.forward_attnres(
                        h,
                        p,
                        list(state_tensors) if state_tensors else None,
                        cache=None,
                        use_cache=False,
                        attnres_block_size=self.attnres_block_size,
                    )
                    return (out, *new_states)

                result = _aster_activation_checkpoint(
                    self.config, custom_forward, hidden, position_ids, *(states or [])
                )
                if torch.is_tensor(result):
                    return result, None
                return result[0], list(result[1:])
            return block.forward_attnres(
                hidden,
                position_ids,
                states,
                cache=cache,
                use_cache=use_cache,
                attnres_block_size=self.attnres_block_size,
            )

        if checkpointed:
            def custom_forward(h: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
                return block(h, p, cache=None, use_cache=False)

            # CRITICAL: return the checkpointed result. The previous implementation
            # discarded it and ran every block a second time without checkpointing.
            return _aster_activation_checkpoint(self.config, custom_forward, hidden, position_ids)
        return block(hidden, position_ids, cache=cache, use_cache=use_cache)


    def _run_block_segment(
        self,
        blocks: list[AsterBlock],
        hidden: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Checkpoint several consecutive blocks behind one saved boundary.

        Per-block checkpointing still retains one [B,T,D] input at every layer.
        At very long context those boundaries alone become multiple GiB. Segmenting
        reduces that storage while keeping exactly the same block equations.
        """
        if not blocks:
            return hidden

        def custom_forward(h: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
            for segment_block in blocks:
                h = segment_block(h, p, cache=None, use_cache=False)
            return h

        # TE's non-reentrant path records saved-tensor hooks throughout the whole
        # callable and cannot early-stop recomputation. Across multiple FP8 blocks
        # that path can retain enough state to oversubscribe a laptop GPU. The
        # reentrant path saves only the segment inputs and is valid here because
        # hidden requires gradients and the segment returns one gradient-bearing
        # tensor. Per-block and tuple-valued AttnRes checkpoints remain
        # non-reentrant.
        return _aster_activation_checkpoint(
            self.config,
            custom_forward,
            hidden,
            position_ids,
            use_reentrant=True,
        )

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
        if self.config.lm_loss_backend == "torch_linear_ce":
            linear_ce = getattr(F, "linear_cross_entropy", None)
            options_cls = getattr(nn, "LinearCrossEntropyOptions", None)
            if linear_ce is None or options_cls is None:
                raise RuntimeError("torch_linear_ce requires PyTorch 2.13+ LinearCrossEntropy")
            options = options_cls(
                chunking_method=self.config.linear_ce_chunking_method,
                acc_policy=self.config.linear_ce_acc_policy,
            )
            return linear_ce(
                hidden.reshape(-1, hidden.shape[-1]),
                self.lm_head.weight,
                labels.reshape(-1),
                reduction="mean",
                ignore_index=ignore_index,
                options=options,
            )

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
        depth_states: list[torch.Tensor] | None = None
        if (
            not self.use_block_attnres
            and self.config.gradient_checkpointing
            and self.training
            and not use_cache
            and self.config.checkpoint_segment_size > 1
        ):
            segment_size = self.config.checkpoint_segment_size
            for start_idx in range(0, len(self.blocks), segment_size):
                segment = list(self.blocks[start_idx : start_idx + segment_size])
                hidden = self._run_block_segment(segment, hidden, position_ids)
        else:
            for block in self.blocks:
                if self.use_block_attnres:
                    hidden, depth_states = self._run_block(
                        block,
                        hidden,
                        position_ids,
                        cache,
                        use_cache,
                        depth_states=depth_states,
                    )
                else:
                    hidden = self._run_block(block, hidden, position_ids, cache, use_cache)

        if self.use_block_attnres:
            assert self.final_attnres is not None
            hidden = self.final_attnres([*(depth_states or []), hidden])
        # Keep the raw backbone state for faithful sequential MTP. The main LM head
        # still consumes the ordinary final-normalized state.
        backbone_hidden = hidden
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
        moe_modules = [
            block.ffn for block in self.blocks
            if isinstance(block.ffn, (DeepSeekStyleMoE, LatentMoE))
        ]
        if (
            self.mtp_deepseek_block is not None
            and isinstance(self.mtp_deepseek_block.ffn, (DeepSeekStyleMoE, LatentMoE))
        ):
            moe_modules.append(self.mtp_deepseek_block.ffn)
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

        if self.config.mtp_depth > 0 and (labels is not None or return_mtp):
            if self.config.mtp_architecture == "low_rank":
                if self.mtp is None:
                    raise RuntimeError("low-rank MTP module was not initialized")
                losses: list[torch.Tensor] = []
                future = hidden
                for head_idx, head in enumerate(self.mtp.heads):
                    if self.config.gradient_checkpointing and self.training:
                        future = _aster_activation_checkpoint(self.config, head, future)
                    else:
                        future = head(future)
                    if return_mtp and mtp_logits is not None:
                        mtp_logits.append(self.lm_head(self.embedding_out_proj(future)))
                    if labels is not None:
                        shift = head_idx + 1
                        if labels.shape[1] > shift:
                            losses.append(
                                self._projected_cross_entropy(
                                    self.embedding_out_proj(future[:, :-shift]),
                                    labels[:, shift:],
                                    ignore_index,
                                )
                            )
                if losses:
                    mtp_loss = torch.stack(losses).mean()
                    total_loss = total_loss + self.config.mtp_loss_weight * mtp_loss
            else:
                if return_mtp and labels is None:
                    raise RuntimeError(
                        "DeepSeek-style MTP drafting is staged: it needs the proposed next-token "
                        "embedding and its own incremental cache. The existing full-prefix reference "
                        "verifier is intentionally not reused as a fake speed path."
                    )
                if labels is not None and labels.shape[1] > 1:
                    assert self.mtp_hnorm is not None
                    assert self.mtp_enorm is not None
                    assert self.mtp_eh_proj is not None
                    assert self.mtp_deepseek_block is not None
                    assert self.mtp_final_norm is not None
                    # At source position i, combine h_i with the actual token t_{i+1}
                    # embedding and predict t_{i+2}. Roll position IDs in lockstep.
                    future_embed = self.embedding_in_proj(self.token_embedding(input_ids[:, 1:]))
                    future_embed = self.embedding_dropout(future_embed)
                    fused = torch.cat(
                        (
                            self.mtp_enorm(future_embed),
                            self.mtp_hnorm(backbone_hidden[:, :-1]),
                        ),
                        dim=-1,
                    )
                    mtp_hidden = self.mtp_eh_proj(fused)
                    mtp_positions = position_ids[:, 1:]
                    if self.config.gradient_checkpointing and self.training:
                        def mtp_forward(h: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
                            return self.mtp_deepseek_block(h, p, cache=None, use_cache=False)
                        mtp_hidden = _aster_activation_checkpoint(
                            self.config, mtp_forward, mtp_hidden, mtp_positions
                        )
                    else:
                        mtp_hidden = self.mtp_deepseek_block(
                            mtp_hidden, mtp_positions, cache=None, use_cache=False
                        )
                    mtp_hidden = self.mtp_final_norm(mtp_hidden)
                    target = labels[:, 1:].clone()
                    # If the main transition h_i -> token_{i+1} is masked (e.g. an
                    # EOS/document boundary), do not let MTP leap across that boundary.
                    target = target.masked_fill(labels[:, :-1].eq(ignore_index), ignore_index)
                    mtp_loss = self._projected_cross_entropy(
                        self.embedding_out_proj(mtp_hidden), target, ignore_index
                    )
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
            if isinstance(block.ffn, (DeepSeekStyleMoE, LatentMoE))
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
    def update_moe_router_biases(self, *, collect_stats: bool = True) -> dict[str, float]:
        loads = []
        biases = []
        candidate_moe = [
            block.ffn for block in self.blocks
            if isinstance(block.ffn, (DeepSeekStyleMoE, LatentMoE))
        ]
        if (
            self.mtp_deepseek_block is not None
            and isinstance(self.mtp_deepseek_block.ffn, (DeepSeekStyleMoE, LatentMoE))
        ):
            candidate_moe.append(self.mtp_deepseek_block.ffn)
        for moe in candidate_moe:
            load = moe.update_routing_bias()
            if load is not None:
                loads.append(load)
                biases.append(moe.routing_bias)
        if not loads or not collect_stats:
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
            if isinstance(block.ffn, (DeepSeekStyleMoE, LatentMoE)):
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
            "kda_num_heads": self.config.kda_num_heads or self.config.n_heads,
            "kda_head_dim": self.config.kda_head_dim or self.config.head_dim,
            "kda_projection_width": (self.config.kda_num_heads or self.config.n_heads)
            * (self.config.kda_head_dim or self.config.head_dim),
            "gdn2_layers": pattern.count("gdn2"),
            "latent_attention_layers": pattern.count("latent"),
            "mtp_depth": self.config.mtp_depth,
            "mtp_architecture": self.config.mtp_architecture,
            "mtp_block_kind": self.config.mtp_block_kind,
            "fla_enabled": self.uses_fla,
            "max_sequence_length": self.config.max_seq_len,
            "ffn_type": self.config.ffn_type,
            "moe_layers": sum(isinstance(block.ffn, (DeepSeekStyleMoE, LatentMoE)) for block in self.blocks),
            "active_parameters_estimate": self.active_parameter_count(),
            "attention_window": self.config.attention_window,
            "latent_cache_width": self.config.latent_rank + self.config.rope_dim,
        }
