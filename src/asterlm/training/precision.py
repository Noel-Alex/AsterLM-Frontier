from __future__ import annotations

from contextlib import ExitStack, nullcontext
from dataclasses import dataclass
from typing import ContextManager

import torch

from asterlm.config import TrainConfig


@dataclass(slots=True)
class PrecisionManager:
    """Owns ordinary AMP and optional Transformer Engine FP8 contexts.

    Transformer Engine only executes FP8 kernels for TE modules. Norms, recurrent
    updates, routing arithmetic, losses, and unsupported operations remain BF16/FP32.
    The context deliberately covers the forward pass only, as required by TE.
    """

    config: TrainConfig
    device: torch.device
    autocast_dtype: torch.dtype

    def __post_init__(self) -> None:
        self._te = None
        self._recipe = None
        if self.config.precision_backend == "transformer_engine_fp8":
            if self.device.type != "cuda":
                raise RuntimeError("Transformer Engine FP8 requires CUDA")
            try:
                import transformer_engine.pytorch as te
                from transformer_engine.common.recipe import DelayedScaling, Format
            except ImportError as exc:  # pragma: no cover - optional CUDA package
                raise ImportError(
                    "precision_backend='transformer_engine_fp8' requires Transformer Engine. "
                    "Install the CUDA/FP8 optional dependencies described in docs/ENVIRONMENT.md."
                ) from exc
            major, minor = torch.cuda.get_device_capability(self.device)
            if (major, minor) < (8, 9):
                raise RuntimeError("Transformer Engine FP8 delayed scaling requires SM89 (Ada) or newer")
            fmt = Format.HYBRID if self.config.fp8_format == "hybrid" else Format.E4M3
            self._te = te
            self._recipe = DelayedScaling(
                fp8_format=fmt,
                amax_history_len=self.config.fp8_amax_history_len,
                amax_compute_algo=self.config.fp8_amax_compute_algo,
            )

    def forward_context(self) -> ContextManager:
        stack = ExitStack()
        if self.device.type == "cuda" and self.autocast_dtype != torch.float32:
            stack.enter_context(torch.autocast(device_type="cuda", dtype=self.autocast_dtype))
        if self._te is not None:
            # New TE versions expose ``autocast``. Retain compatibility with older
            # releases that called the same API ``fp8_autocast``.
            fn = getattr(self._te, "autocast", None) or getattr(self._te, "fp8_autocast")
            try:
                stack.enter_context(fn(enabled=True, recipe=self._recipe))
            except TypeError:  # older keyword name
                stack.enter_context(fn(enabled=True, fp8_recipe=self._recipe))
        return stack

    def activation_context(self) -> ContextManager:
        if not self.config.activation_offload or self.device.type != "cuda":
            return nullcontext()
        # save_on_cpu is exact: activations are copied back before their backward use.
        # It trades PCIe bandwidth and host RAM for VRAM, matching the project's
        # "must fit, may run slowly" objective.
        return torch.autograd.graph.save_on_cpu(
            pin_memory=self.config.activation_offload_pin_memory,
            device_type="cuda",
        )
