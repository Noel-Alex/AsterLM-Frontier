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
    except (ImportError, ModuleNotFoundError):
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
        self.n_heads = config.kda_num_heads or config.n_heads
        self.head_dim = config.kda_head_dim or config.head_dim
        self.value_head_dim = int(self.head_dim * config.kda_expand_v)
        if self.value_head_dim <= 0 or self.value_head_dim != self.head_dim * config.kda_expand_v:
            raise ValueError("kda_expand_v must produce an integer value-head dimension")
        self.key_dim = self.n_heads * self.head_dim
        self.value_dim = self.n_heads * self.value_head_dim
        self.lower_bound = config.kda_lower_bound
        self.allow_negative = config.kda_allow_negative_eigenvalues

        self.q_proj = nn.Linear(d, self.key_dim, bias=False)
        self.k_proj = nn.Linear(d, self.key_dim, bias=False)
        self.v_proj = nn.Linear(d, self.value_dim, bias=False)
        self.decay_proj = nn.Linear(d, self.key_dim, bias=True)
        self.beta_proj = nn.Linear(d, self.n_heads, bias=True)
        self.sign_proj = nn.Linear(d, self.key_dim, bias=True) if self.allow_negative else None
        self.out_gate = nn.Linear(d, self.value_dim, bias=True)
        self.out_norm = HeadRMSNorm(self.value_head_dim, config.rms_eps)
        self.out_proj = nn.Linear(self.value_dim, d, bias=False)
        self.out_proj._is_residual_projection = True

    def forward(
        self, hidden: torch.Tensor, state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = hidden.shape
        q = self.q_proj(hidden).view(bsz, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(hidden).view_as(q)
        v = self.v_proj(hidden).view(
            bsz, seq_len, self.n_heads, self.value_head_dim
        )
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
                self.value_head_dim,
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
        out = self.out_proj(out.reshape(bsz, seq_len, self.value_dim))
        return out, state


class KDA(nn.Module):
    def __init__(self, config: AsterConfig, kda_idx: int) -> None:
        super().__init__()
        self.kda_idx = kda_idx
        self._d_model = config.d_model
        self._n_heads = config.kda_num_heads or config.n_heads
        self._head_dim = config.kda_head_dim or config.head_dim
        self._value_head_dim = int(self._head_dim * config.kda_expand_v)
        self._short_conv = config.kda_short_conv
        self._conv_size = config.kda_conv_size
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
                head_dim=config.kda_head_dim or config.head_dim,
                num_heads=config.kda_num_heads or config.n_heads,
                num_v_heads=config.kda_num_heads or config.n_heads,
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

    def logical_parameter_count(self) -> int:
        """Return the production KDA geometry independent of the execution backend.

        The readable Torch recurrence intentionally uses larger dense decay and gate
        projections than FLA's KimiDeltaAttention.  It is a numerical-control backend,
        not a different model candidate, so architecture/FLOP accounting must describe
        the production KDA parameterization even when FLA is unavailable in CPU CI.
        """

        d = self._d_model
        h = self._n_heads
        k = self._head_dim
        v = self._value_head_dim
        key_width = h * k
        value_width = h * v

        total = h + key_width  # A_log and dt_bias
        total += (2 * key_width + value_width) * d  # q, k, v projections
        if self._short_conv:
            total += (2 * key_width + value_width) * self._conv_size
        total += k * d + key_width * k  # low-rank f projection
        total += h * d  # beta projection
        total += k * d + value_width * k + value_width  # low-rank output gate
        total += v  # per-head output norm
        total += d * value_width  # output projection
        return total

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
