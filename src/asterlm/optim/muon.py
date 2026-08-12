from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
from torch.optim import Optimizer


@dataclass(frozen=True)
class _MuonJob:
    parameter: torch.nn.Parameter
    gradient: torch.Tensor
    momentum_buffer: torch.Tensor


def _megabatch_job_limit(
    shape: torch.Size,
    split_count: int,
    workspace_bytes: int,
) -> int:
    """Conservatively bound temporary NS storage for a group of parameters."""

    rows, columns = int(shape[0]) // split_count, int(shape[1])
    # Stacked FP32 inputs/results, BF16 working matrices and Gram/correction
    # intermediates. This intentionally overestimates the portable PyTorch path.
    bytes_per_matrix = 16 * rows * columns + 8 * min(rows, columns) ** 2
    return max(1, workspace_bytes // max(bytes_per_matrix * split_count, 1))


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


def zeropower_via_newton_schulz5_batched(
    matrices: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Batched polar-factor approximation for equal-shaped per-head matrices.

    Kimi K3 partitions Q/K/V momentum by head. Launching every Newton--Schulz
    multiply in a Python head loop fragments the optimizer into many tiny kernels;
    a leading batch dimension preserves the independent per-head mathematics while
    allowing PyTorch/CUDA to issue batched matrix multiplies.
    """

    if matrices.ndim != 3:
        raise ValueError("Batched Muon orthogonalization expects [blocks, rows, cols]")
    original_dtype = matrices.dtype
    x = matrices
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = x.transpose(-2, -1)
    working_dtype = torch.bfloat16 if x.device.type == "cuda" else torch.float32
    x = x.to(working_dtype)
    norms = torch.linalg.vector_norm(x.float(), dim=(-2, -1), keepdim=True)
    x = x / (norms + eps).to(x.dtype)

    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        gram = x @ x.transpose(-2, -1)
        correction = b * gram + c * (gram @ gram)
        x = a * x + correction @ x
    if transposed:
        x = x.transpose(-2, -1)
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
        megabatch: bool = True,
        megabatch_max_gib: float = 0.5,
    ) -> None:
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "ns_steps": ns_steps,
            "nesterov": nesterov,
            "update_rms": update_rms,
            "split_count": 1,
            "split_axis": 0,
            "megabatch": megabatch,
            "megabatch_max_gib": megabatch_max_gib,
        }
        super().__init__(params, defaults)
        self._diagnostics_enabled = False
        self._latest_diagnostics: dict[str, torch.Tensor | float] = {}

    def set_diagnostics_enabled(self, enabled: bool) -> None:
        self._diagnostics_enabled = bool(enabled)

    def diagnostics(self) -> dict[str, float]:
        return {
            name: float(value.detach()) if isinstance(value, torch.Tensor) else float(value)
            for name, value in self._latest_diagnostics.items()
        }

    @torch.no_grad()
    def step(self, closure=None):
        use_megabatch = any(
            bool(group.get("megabatch", self.defaults["megabatch"]))
            for group in self.param_groups
        )
        if use_megabatch:
            return self._step_megabatched(closure)
        return self._step_parameterwise(closure)

    @torch.no_grad()
    def _step_parameterwise(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        diagnostic_sums: dict[str, torch.Tensor] = {}
        diagnostic_counts: dict[str, int] = {}

        def accumulate(name: str, value: torch.Tensor) -> None:
            squared = value.float().square().sum()
            diagnostic_sums[name] = diagnostic_sums.get(name, torch.zeros_like(squared)) + squared
            diagnostic_counts[name] = diagnostic_counts.get(name, 0) + value.numel()

        relative_update_sum: torch.Tensor | None = None
        relative_update_count = 0
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            wd = group["weight_decay"]
            ns_steps = group["ns_steps"]
            nesterov = group["nesterov"]
            target_rms = group["update_rms"]
            split_count = int(group.get("split_count", 1))
            split_axis = int(group.get("split_axis", 0))
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
                if split_count > 1:
                    if split_axis != 0:
                        raise RuntimeError(
                            "Batched per-head Muon currently requires split_axis=0"
                        )
                    if param.shape[split_axis] % split_count:
                        raise RuntimeError(
                            f"Per-head Muon cannot split shape {tuple(param.shape)} "
                            f"into {split_count} blocks on axis {split_axis}"
                        )
                    block_shape = (
                        split_count,
                        param.shape[0] // split_count,
                        param.shape[1],
                    )
                    blocks = update.reshape(block_shape)
                    update = zeropower_via_newton_schulz5_batched(
                        blocks, steps=ns_steps
                    )
                    block_scale = target_rms * math.sqrt(max(block_shape[1:]))
                    update = update.mul(block_scale).reshape_as(param)
                else:
                    update = zeropower_via_newton_schulz5(update, steps=ns_steps)
                    # Match the update RMS convention used in Kimi K2's Muon recipe.
                    scale = target_rms * math.sqrt(max(param.shape))
                    update = update.mul(scale)
                if self._diagnostics_enabled:
                    accumulate("muon_momentum", buffer)
                    accumulate("muon_update", update)
                    parameter_rms = param.float().square().mean().sqrt()
                    applied_rms = update.float().square().mean().sqrt() * lr
                    relative = applied_rms / parameter_rms.clamp_min(1e-12)
                    relative_update_sum = (
                        relative
                        if relative_update_sum is None
                        else relative_update_sum + relative
                    )
                    relative_update_count += 1
                if wd:
                    param.mul_(1.0 - lr * wd)
                param.add_(update.to(param.dtype), alpha=-lr)
        if self._diagnostics_enabled:
            self._latest_diagnostics = {
                f"{name}_global_rms": (
                    total / max(diagnostic_counts[name], 1)
                ).sqrt()
                for name, total in diagnostic_sums.items()
            }
            if relative_update_sum is not None:
                self._latest_diagnostics["muon_relative_update_rms_mean"] = (
                    relative_update_sum / max(relative_update_count, 1)
                )
            self._latest_diagnostics["muon_matrix_count"] = float(relative_update_count)
        else:
            self._latest_diagnostics = {}
        return loss

    @torch.no_grad()
    def _step_megabatched(self, closure=None):
        """Execute identical Muon math in bounded, equal-shaped matrix batches."""

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        buckets: dict[tuple[Any, ...], list[_MuonJob]] = defaultdict(list)
        for group in self.param_groups:
            lr = float(group["lr"])
            momentum = float(group["momentum"])
            weight_decay = float(group["weight_decay"])
            ns_steps = int(group["ns_steps"])
            nesterov = bool(group["nesterov"])
            target_rms = float(group["update_rms"])
            split_count = int(group.get("split_count", 1))
            split_axis = int(group.get("split_axis", 0))
            workspace_gib = float(
                group.get("megabatch_max_gib", self.defaults["megabatch_max_gib"])
            )
            if split_axis != 0:
                raise RuntimeError("Batched Muon currently requires split_axis=0")
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")
                if parameter.ndim != 2:
                    raise RuntimeError(
                        "Muon received a non-matrix parameter with shape "
                        f"{tuple(parameter.shape)}"
                    )
                if parameter.shape[0] % split_count:
                    raise RuntimeError(
                        f"Per-head Muon cannot split shape {tuple(parameter.shape)} "
                        f"into {split_count} blocks on axis 0"
                    )
                state = self.state[parameter]
                if not state:
                    state["momentum_buffer"] = torch.zeros_like(
                        parameter, dtype=torch.float32
                    )
                key = (
                    parameter.device,
                    parameter.dtype,
                    tuple(parameter.shape),
                    lr,
                    momentum,
                    weight_decay,
                    ns_steps,
                    nesterov,
                    target_rms,
                    split_count,
                    workspace_gib,
                )
                buckets[key].append(
                    _MuonJob(parameter, gradient, state["momentum_buffer"])
                )

        diagnostic_sums: dict[str, torch.Tensor] = {}
        diagnostic_counts: dict[str, int] = {}
        relative_update_sum: torch.Tensor | None = None
        relative_update_count = 0
        chunk_count = 0
        max_matrices_per_chunk = 0

        def accumulate(name: str, value: torch.Tensor) -> None:
            squared = value.float().square().sum()
            diagnostic_sums[name] = (
                diagnostic_sums.get(name, torch.zeros_like(squared)) + squared
            )
            diagnostic_counts[name] = diagnostic_counts.get(name, 0) + value.numel()

        for key, jobs in buckets.items():
            (
                _device,
                _dtype,
                shape_tuple,
                lr,
                momentum,
                weight_decay,
                ns_steps,
                nesterov,
                target_rms,
                split_count,
                workspace_gib,
            ) = key
            shape = torch.Size(shape_tuple)
            limit = _megabatch_job_limit(
                shape,
                split_count,
                max(1, int(workspace_gib * 2**30)),
            )
            block_rows = shape[0] // split_count
            block_scale = target_rms * math.sqrt(max(block_rows, shape[1]))
            for start in range(0, len(jobs), limit):
                chunk = jobs[start : start + limit]
                parameters = [job.parameter for job in chunk]
                gradients = [job.gradient.float() for job in chunk]
                buffers = [job.momentum_buffer for job in chunk]
                torch._foreach_mul_(buffers, momentum)
                torch._foreach_add_(buffers, gradients)
                if nesterov:
                    updates = torch._foreach_mul(buffers, momentum)
                    torch._foreach_add_(updates, gradients)
                else:
                    updates = buffers
                matrices = torch.cat(
                    [
                        update.reshape(split_count, block_rows, shape[1])
                        for update in updates
                    ],
                    dim=0,
                )
                orthogonal = zeropower_via_newton_schulz5_batched(
                    matrices, steps=ns_steps
                ).mul_(block_scale)
                parameter_updates = list(
                    orthogonal.reshape(len(chunk), *shape).unbind(0)
                )
                if self._diagnostics_enabled:
                    for parameter, buffer, update in zip(
                        parameters, buffers, parameter_updates, strict=True
                    ):
                        accumulate("muon_momentum", buffer)
                        accumulate("muon_update", update)
                        parameter_rms = parameter.float().square().mean().sqrt()
                        applied_rms = update.float().square().mean().sqrt() * lr
                        relative = applied_rms / parameter_rms.clamp_min(1e-12)
                        relative_update_sum = (
                            relative
                            if relative_update_sum is None
                            else relative_update_sum + relative
                        )
                        relative_update_count += 1
                if weight_decay:
                    torch._foreach_mul_(parameters, 1.0 - lr * weight_decay)
                torch._foreach_add_(parameters, parameter_updates, alpha=-lr)
                chunk_count += 1
                max_matrices_per_chunk = max(
                    max_matrices_per_chunk, len(chunk) * split_count
                )

        if self._diagnostics_enabled:
            self._latest_diagnostics = {
                f"{name}_global_rms": (
                    total / max(diagnostic_counts[name], 1)
                ).sqrt()
                for name, total in diagnostic_sums.items()
            }
            if relative_update_sum is not None:
                self._latest_diagnostics["muon_relative_update_rms_mean"] = (
                    relative_update_sum / max(relative_update_count, 1)
                )
            self._latest_diagnostics.update(
                {
                    "muon_matrix_count": float(relative_update_count),
                    "muon_megabatch_bucket_count": float(len(buckets)),
                    "muon_megabatch_chunk_count": float(chunk_count),
                    "muon_megabatch_max_matrices": float(max_matrices_per_chunk),
                }
            )
        else:
            self._latest_diagnostics = {}
        return loss
