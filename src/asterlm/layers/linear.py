from __future__ import annotations

from torch import nn


def build_linear(
    in_features: int,
    out_features: int,
    *,
    bias: bool,
    backend: str = "torch",
    loqt_rank: int = 32,
    loqt_alpha: float = 32.0,
    loqt_group_size: int = 64,
    init_std: float = 0.02,
) -> nn.Module:
    """Construct a linear layer for BF16, Transformer Engine FP8, or LoQT-style INT4."""
    if backend == "torch":
        layer = nn.Linear(in_features, out_features, bias=bias)
        layer._aster_linear = True  # type: ignore[attr-defined]
        layer._aster_linear_backend = backend  # type: ignore[attr-defined]
        return layer
    if backend == "loqt_int4":
        from asterlm.quantization.loqt import LoQTLinear

        return LoQTLinear(
            in_features,
            out_features,
            bias=bias,
            rank=loqt_rank,
            alpha=loqt_alpha,
            group_size=loqt_group_size,
            init_std=init_std,
        )
    if backend != "transformer_engine":
        raise ValueError(f"unknown linear backend: {backend}")
    try:
        import transformer_engine.pytorch as te
    except ImportError as exc:  # pragma: no cover - optional CUDA dependency
        raise ImportError(
            "linear_backend='transformer_engine' requires NVIDIA Transformer Engine. "
            "Install the CUDA extra described in docs/ENVIRONMENT.md."
        ) from exc
    layer = te.Linear(in_features, out_features, bias=bias)
    layer._aster_linear = True  # type: ignore[attr-defined]
    layer._aster_linear_backend = backend  # type: ignore[attr-defined]
    return layer


def mark_residual(module: nn.Module) -> nn.Module:
    module._is_residual_projection = True  # type: ignore[attr-defined]
    return module
