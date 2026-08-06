from __future__ import annotations

import warnings

import torch
from torch import nn
from torch.nn import functional as F

from asterlm.cache import AsterCache
from asterlm.config import AsterConfig
from .norm import HeadRMSNorm


def fla_is_available() -> bool:
    try:
        from fla.layers.kda import KimiDeltaAttention  # noqa: F401

        return True
    except Exception:
        return False


class TorchGatedDeltaNet(nn.Module):
    """Readable recurrent fallback for KDA-like gated delta updates.

    This is deliberately not advertised as a speed kernel. It provides correct tensor
    semantics, gradients, cache behavior, and CPU smoke tests. Install ``fla-core`` on
    Linux/WSL2 for actual training and decoding throughput.
    """

    def __init__(self, config: AsterConfig) -> None:
        super().__init__()
        d = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.lower_bound = config.kda_lower_bound
        self.allow_negative = config.kda_allow_negative_eigenvalues

        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.decay_proj = nn.Linear(d, d, bias=True)
        self.beta_proj = nn.Linear(d, self.n_heads, bias=True)
        self.sign_proj = nn.Linear(d, d, bias=True) if self.allow_negative else None
        self.out_gate = nn.Linear(d, d, bias=True)
        self.out_norm = HeadRMSNorm(self.head_dim, config.rms_eps)
        self.out_proj = nn.Linear(d, d, bias=False)
        self.out_proj._is_residual_projection = True

    def forward(
        self, hidden: torch.Tensor, state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = hidden.shape
        q = self.q_proj(hidden).view(bsz, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(hidden).view_as(q)
        v = self.v_proj(hidden).view_as(q)
        q = F.normalize(q.float(), dim=-1).to(hidden.dtype)
        k = F.normalize(k.float(), dim=-1).to(hidden.dtype)

        log_decay = -F.softplus(self.decay_proj(hidden).float())
        if self.lower_bound is not None:
            log_decay = log_decay.clamp_min(self.lower_bound)
        decay = log_decay.exp().to(hidden.dtype).view_as(q)
        if self.sign_proj is not None:
            decay = decay * torch.tanh(self.sign_proj(hidden)).view_as(q)
        beta = torch.sigmoid(self.beta_proj(hidden)).to(hidden.dtype)

        if state is None:
            state = torch.zeros(
                bsz,
                self.n_heads,
                self.head_dim,
                self.head_dim,
                dtype=hidden.dtype,
                device=hidden.device,
            )

        outputs: list[torch.Tensor] = []
        for t in range(seq_len):
            qt, kt, vt = q[:, t], k[:, t], v[:, t]
            state = state * decay[:, t].unsqueeze(-1)
            predicted = torch.einsum("bhd,bhdv->bhv", kt, state)
            delta = (vt - predicted) * beta[:, t].unsqueeze(-1)
            state = state + torch.einsum("bhd,bhv->bhdv", kt, delta)
            outputs.append(torch.einsum("bhd,bhdv->bhv", qt, state))

        out = torch.stack(outputs, dim=1)
        gate = torch.sigmoid(self.out_gate(hidden)).view_as(out)
        out = self.out_norm(out) * gate
        out = self.out_proj(out.reshape(bsz, seq_len, -1))
        return out, state


class KDA(nn.Module):
    def __init__(self, config: AsterConfig, kda_idx: int) -> None:
        super().__init__()
        self.kda_idx = kda_idx
        requested = config.kda_backend
        available = fla_is_available()
        self.uses_fla = requested == "fla" or (requested == "auto" and available)
        if requested == "fla" and not available:
            raise ImportError("kda_backend='fla' requires `pip install fla-core transformers`")

        if self.uses_fla:
            from fla.layers.kda import KimiDeltaAttention

            self.impl = KimiDeltaAttention(
                hidden_size=config.d_model,
                expand_v=config.kda_expand_v,
                head_dim=config.head_dim,
                num_heads=config.n_heads,
                num_v_heads=config.n_heads,
                mode="chunk",
                use_short_conv=config.kda_short_conv,
                allow_neg_eigval=config.kda_allow_negative_eigenvalues,
                safe_gate=config.kda_safe_gate,
                lower_bound=config.kda_lower_bound,
                conv_size=config.kda_conv_size,
                layer_idx=kda_idx,
                norm_eps=config.rms_eps,
            )
        else:
            if requested == "auto":
                warnings.warn(
                    "fla-core is unavailable; using the slow PyTorch gated-delta fallback. "
                    "This is suitable for tests, not serious training.",
                    stacklevel=2,
                )
            self.impl = TorchGatedDeltaNet(config)

    def forward(
        self,
        hidden: torch.Tensor,
        cache: AsterCache | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        if self.uses_fla:
            fla_cache = None if cache is None else cache.fla_cache
            out, _, _ = self.impl(
                hidden,
                past_key_values=fla_cache,
                use_cache=use_cache,
            )
            return out

        state = None if cache is None else cache.kda_states.get(self.kda_idx)
        out, new_state = self.impl(hidden, state)
        if use_cache and cache is not None:
            cache.kda_states[self.kda_idx] = new_state.detach()
        return out
