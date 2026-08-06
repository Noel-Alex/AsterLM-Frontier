from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch.optim import Optimizer


def zeropower_via_newton_schulz5(matrix: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximate the polar factor used by Muon with a stable quintic iteration.

    Computation runs in bfloat16 where supported, while normalization is computed in
    float32. The result is returned in the input dtype.
    """

    if matrix.ndim != 2:
        raise ValueError("Muon orthogonalization expects a matrix")
    original_dtype = matrix.dtype
    x = matrix
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    working_dtype = torch.bfloat16 if x.device.type == "cuda" else torch.float32
    x = x.to(working_dtype)
    x = x / (x.float().norm() + eps).to(x.dtype)

    # Coefficients popularized by the public Muon reference implementation.
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        gram = x @ x.T
        correction = b * gram + c * (gram @ gram)
        x = a * x + correction @ x
    if transposed:
        x = x.T
    return x.to(original_dtype)


class Muon(Optimizer):
    """Single-device Muon optimizer for hidden 2-D parameter matrices.

    Embeddings, scalar/vector parameters, convolution kernels, and output heads should
    be handled by AdamW. ``build_hybrid_optimizer`` performs that partition.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.01,
        momentum: float = 0.95,
        weight_decay: float = 0.1,
        ns_steps: int = 5,
        nesterov: bool = True,
        update_rms: float = 0.2,
    ) -> None:
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            nesterov=nesterov,
            update_rms=update_rms,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            wd = group["weight_decay"]
            ns_steps = group["ns_steps"]
            nesterov = group["nesterov"]
            target_rms = group["update_rms"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")
                if param.ndim != 2:
                    raise RuntimeError(f"Muon received a non-matrix parameter with shape {tuple(param.shape)}")

                state = self.state[param]
                if not state:
                    state["momentum_buffer"] = torch.zeros_like(param, dtype=torch.float32)
                buffer = state["momentum_buffer"]
                buffer.mul_(momentum).add_(grad.float())
                update = grad.float().add(buffer, alpha=momentum) if nesterov else buffer
                update = zeropower_via_newton_schulz5(update, steps=ns_steps)

                # Match the update RMS convention used in Kimi K2's Muon recipe.
                scale = target_rms * math.sqrt(max(param.shape))
                update = update.mul(scale)
                if wd:
                    param.mul_(1.0 - lr * wd)
                param.add_(update.to(param.dtype), alpha=-lr)
        return loss
