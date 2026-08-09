from __future__ import annotations

import torch
from torch import nn

from asterlm.cache import AsterCache
from asterlm.config import AsterConfig


def gdn2_is_available() -> bool:
    try:
        from fla.layers.gdn2 import GatedDeltaNet2  # noqa: F401

        return True
    except Exception:
        return False


class GDN2(nn.Module):
    """Aster wrapper around FLA's Gated DeltaNet-2 layer.

    The wrapper intentionally uses FLA as an external dependency instead of copying
    NVIDIA's non-commercial reference implementation into Aster. FLA exposes its own
    adapted implementation and cache interface. The recurrent index shares the same
    LegacyFLACache namespace as KDA layers, allowing KDA/GDN2 hybrid patterns.
    """

    def __init__(self, config: AsterConfig, recurrent_idx: int) -> None:
        super().__init__()
        if not gdn2_is_available():
            raise ImportError(
                "GDN2 requires a recent full `flash-linear-attention` build with "
                "`fla.layers.gdn2.GatedDeltaNet2`."
            )
        from fla.layers.gdn2 import GatedDeltaNet2

        self.recurrent_idx = int(recurrent_idx)
        self.uses_fla = True
        self.impl = GatedDeltaNet2(
            hidden_size=config.d_model,
            expand_v=config.kda_expand_v,
            head_dim=config.head_dim,
            num_heads=config.n_heads,
            num_v_heads=config.n_heads,
            mode="chunk",
            use_short_conv=config.kda_short_conv,
            allow_neg_eigval=config.kda_allow_negative_eigenvalues,
            conv_size=config.kda_conv_size,
            layer_idx=self.recurrent_idx,
            norm_eps=config.rms_eps,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        cache: AsterCache | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        fla_cache = None if cache is None else cache.fla_cache
        out, _, _ = self.impl(
            hidden,
            past_key_values=fla_cache,
            use_cache=use_cache,
        )
        return out
