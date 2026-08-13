from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
from torch.optim import Optimizer


@dataclass
class _MuonJob:
    parameter: torch.nn.Parameter
    gradient: torch.Tensor | None
    state: dict[str, Any]


def quantize_blockwise_int8_(
    tensor: torch.Tensor,
    block_size: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a floating tensor with independent symmetric INT8 blocks.

    The input is used as scratch space and is intentionally destroyed. This avoids
    retaining another parameter-sized FP32 temporary at the optimizer peak. The
    scheme follows the published 8-bit Muon state recipe: one abs-max scale per
    2,048 values and signed linear quantization in [-127, 127].
    """

    if not tensor.is_floating_point():
        raise TypeError("blockwise INT8 quantization requires a floating tensor")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    flat = tensor.reshape(-1)
    full_values = (flat.numel() // block_size) * block_size
    scale_parts: list[torch.Tensor] = []
    if full_values:
        blocks = flat[:full_values].view(-1, block_size)
        scales = blocks.abs().amax(dim=1).div_(127.0)
        divisor = scales.clamp_min(torch.finfo(scales.dtype).tiny)
        blocks.div_(divisor.unsqueeze(1)).round_().clamp_(-127, 127)
        scale_parts.append(scales)
    if full_values < flat.numel():
        tail = flat[full_values:]
        tail_scale = tail.abs().amax().reshape(1).div_(127.0)
        divisor = tail_scale.clamp_min(torch.finfo(tail_scale.dtype).tiny)
        tail.div_(divisor).round_().clamp_(-127, 127)
        scale_parts.append(tail_scale)
    scales = torch.cat(scale_parts) if scale_parts else tensor.new_empty((0,))
    return tensor.to(torch.int8), scales


def dequantize_blockwise_int8(
    quantized: torch.Tensor,
    scales: torch.Tensor,
    block_size: int = 2048,
) -> torch.Tensor:
    """Restore an INT8 blockwise tensor into FP32 optimizer workspace."""

    if quantized.dtype != torch.int8:
        raise TypeError("blockwise INT8 state must use torch.int8")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    output = quantized.float()
    flat = output.reshape(-1)
    full_blocks = flat.numel() // block_size
    if full_blocks:
        flat[: full_blocks * block_size].view(full_blocks, block_size).mul_(
            scales[:full_blocks].unsqueeze(1)
        )
    if full_blocks * block_size < flat.numel():
        flat[full_blocks * block_size :].mul_(scales[full_blocks])
    return output


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
        release_gradients_after_step: bool = False,
        state_dtype: str = "float32",
        quant_block_size: int = 2048,
    ) -> None:
        if state_dtype not in {"float32", "int8_blockwise"}:
            raise ValueError("Muon state_dtype must be float32 or int8_blockwise")
        if quant_block_size <= 0:
            raise ValueError("Muon quant_block_size must be positive")
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
            "release_gradients_after_step": release_gradients_after_step,
            "state_dtype": state_dtype,
            "quant_block_size": quant_block_size,
        }
        super().__init__(params, defaults)
        self._diagnostics_enabled = False
        self._latest_diagnostics: dict[str, torch.Tensor | float] = {}

    @staticmethod
    def _load_momentum(
        state: dict[str, Any],
        parameter: torch.Tensor,
        state_dtype: str,
        block_size: int,
    ) -> torch.Tensor:
        if state_dtype == "float32":
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(parameter, dtype=torch.float32)
            return state["momentum_buffer"]
        if "momentum_q" not in state:
            state["momentum_q"] = torch.zeros_like(parameter, dtype=torch.int8)
            blocks = math.ceil(parameter.numel() / block_size)
            state["momentum_scale"] = torch.zeros(
                blocks, device=parameter.device, dtype=torch.float32
            )
        return dequantize_blockwise_int8(
            state["momentum_q"], state["momentum_scale"], block_size
        )

    @staticmethod
    def _store_momentum(
        state: dict[str, Any],
        buffer: torch.Tensor,
        state_dtype: str,
        block_size: int,
    ) -> None:
        if state_dtype == "float32":
            return
        quantized, scales = quantize_blockwise_int8_(buffer, block_size)
        state["momentum_q"] = quantized
        state["momentum_scale"] = scales

    def set_diagnostics_enabled(self, enabled: bool) -> None:
        self._diagnostics_enabled = bool(enabled)

    def diagnostics(self) -> dict[str, float]:
        return {
            name: float(value.detach()) if isinstance(value, torch.Tensor) else float(value)
            for name, value in self._latest_diagnostics.items()
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore optimizer state without losing the low-bit storage contract.

        PyTorch's generic optimizer loader casts floating-point-compatible state to
        each parameter's dtype. INT8 Muon momentum must remain INT8 instead of being
        silently expanded to BF16/FP32 after a resume.
        """

        raw_low_bit = {
            saved_id: {
                name: saved_state[name]
                for name in ("momentum_q", "momentum_scale")
                if name in saved_state
            }
            for saved_id, saved_state in state_dict.get("state", {}).items()
            if "momentum_q" in saved_state
        }
        super().load_state_dict(state_dict)
        for live_group, saved_group in zip(
            self.param_groups, state_dict.get("param_groups", []), strict=True
        ):
            for parameter, saved_id in zip(
                live_group["params"], saved_group["params"], strict=True
            ):
                raw = raw_low_bit.get(saved_id)
                if raw is None:
                    continue
                state = self.state[parameter]
                state["momentum_q"] = raw["momentum_q"].to(
                    device=parameter.device, dtype=torch.int8
                )
                state["momentum_scale"] = raw["momentum_scale"].to(
                    device=parameter.device, dtype=torch.float32
                )

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
            state_dtype = str(group.get("state_dtype", self.defaults["state_dtype"]))
            release_gradients = bool(
                group.get(
                    "release_gradients_after_step",
                    self.defaults["release_gradients_after_step"],
                )
            )
            quant_block_size = int(
                group.get("quant_block_size", self.defaults["quant_block_size"])
            )
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
                buffer = self._load_momentum(
                    state, param, state_dtype, quant_block_size
                )
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
                self._store_momentum(
                    state, buffer, state_dtype, quant_block_size
                )
                if release_gradients:
                    param.grad = None
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
            state_dtype = str(group.get("state_dtype", self.defaults["state_dtype"]))
            release_gradients = bool(
                group.get(
                    "release_gradients_after_step",
                    self.defaults["release_gradients_after_step"],
                )
            )
            quant_block_size = int(
                group.get("quant_block_size", self.defaults["quant_block_size"])
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
                    state_dtype,
                    quant_block_size,
                    release_gradients,
                )
                buckets[key].append(
                    _MuonJob(parameter, gradient, state)
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
                state_dtype,
                quant_block_size,
                release_gradients,
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
                gradients = [
                    job.gradient.float()
                    for job in chunk
                    if job.gradient is not None
                ]
                if len(gradients) != len(chunk):
                    raise RuntimeError("Muon gradient was released before its update")
                buffers = [
                    self._load_momentum(
                        job.state, job.parameter, state_dtype, quant_block_size
                    )
                    for job in chunk
                ]
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
                for job, buffer in zip(chunk, buffers, strict=True):
                    self._store_momentum(
                        job.state, buffer, state_dtype, quant_block_size
                    )
                    if release_gradients:
                        job.parameter.grad = None
                        job.gradient = None
                if release_gradients:
                    del gradients, updates, matrices, orthogonal, parameter_updates
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
