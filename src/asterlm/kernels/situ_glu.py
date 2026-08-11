from __future__ import annotations

import os

import torch

try:  # Triton is unavailable in native Windows PyTorch environments.
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:  # pragma: no cover - platform/package dependent
    triton = None
    tl = None
    libdevice = None


_SITU_GLU_BACKEND = os.environ.get("ASTER_SITU_GLU_BACKEND", "auto").strip().lower()
if _SITU_GLU_BACKEND not in {"auto", "reference", "triton"}:
    raise ValueError(
        "ASTER_SITU_GLU_BACKEND must be one of: auto, reference, triton"
    )


def configured_situ_glu_backend() -> str:
    """Return the process-pinned SiTU execution treatment for manifests."""

    return _SITU_GLU_BACKEND


def situ_glu_reference(
    gate: torch.Tensor,
    up: torch.Tensor,
    beta_gate: float = 4.0,
    beta_up: float = 25.0,
) -> torch.Tensor:
    """Portable Kimi K3 SiTU-GLU oracle."""

    bounded_gate = beta_gate * torch.tanh(gate / beta_gate)
    bounded_up = beta_up * torch.tanh(up / beta_up)
    return bounded_gate * torch.sigmoid(gate) * bounded_up


if triton is not None:

    @triton.jit
    def _situ_glu_forward(
        gate,
        up,
        output,
        elements: tl.constexpr,
        beta_gate: tl.constexpr,
        beta_up: tl.constexpr,
        block: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block + tl.arange(0, block)
        mask = offsets < elements
        gate_values = tl.load(gate + offsets, mask=mask, other=0.0).to(tl.float32)
        up_values = tl.load(up + offsets, mask=mask, other=0.0).to(tl.float32)
        gate_tanh = libdevice.tanh(gate_values / beta_gate)
        up_tanh = libdevice.tanh(up_values / beta_up)
        output_values = (
            (beta_gate * gate_tanh)
            * tl.sigmoid(gate_values)
            * (beta_up * up_tanh)
        )
        tl.store(output + offsets, output_values, mask=mask)

    @triton.jit
    def _situ_glu_backward(
        gate,
        up,
        output_gradient,
        gate_gradient,
        up_gradient,
        elements: tl.constexpr,
        beta_gate: tl.constexpr,
        beta_up: tl.constexpr,
        block: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block + tl.arange(0, block)
        mask = offsets < elements
        gate_values = tl.load(gate + offsets, mask=mask, other=0.0).to(tl.float32)
        up_values = tl.load(up + offsets, mask=mask, other=0.0).to(tl.float32)
        gradient = tl.load(output_gradient + offsets, mask=mask, other=0.0).to(
            tl.float32
        )

        gate_tanh = libdevice.tanh(gate_values / beta_gate)
        up_tanh = libdevice.tanh(up_values / beta_up)
        bounded_gate = beta_gate * gate_tanh
        bounded_up = beta_up * up_tanh
        sigmoid_gate = tl.sigmoid(gate_values)
        bounded_gate_gradient = 1.0 - gate_tanh * gate_tanh
        bounded_up_gradient = 1.0 - up_tanh * up_tanh
        sigmoid_gradient = sigmoid_gate * (1.0 - sigmoid_gate)

        dgate = gradient * bounded_up * (
            bounded_gate_gradient * sigmoid_gate
            + bounded_gate * sigmoid_gradient
        )
        dup = gradient * bounded_gate * sigmoid_gate * bounded_up_gradient
        tl.store(gate_gradient + offsets, dgate, mask=mask)
        tl.store(up_gradient + offsets, dup, mask=mask)


class _TritonSiTUGLU(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        gate: torch.Tensor,
        up: torch.Tensor,
        beta_gate: float,
        beta_up: float,
    ) -> torch.Tensor:
        gate_contiguous = gate.contiguous()
        up_contiguous = up.contiguous()
        output = torch.empty_like(gate_contiguous)
        elements = gate_contiguous.numel()
        block = 256
        _situ_glu_forward[(triton.cdiv(elements, block),)](
            gate_contiguous,
            up_contiguous,
            output,
            elements=elements,
            beta_gate=beta_gate,
            beta_up=beta_up,
            block=block,
            num_warps=4,
        )
        ctx.save_for_backward(gate_contiguous, up_contiguous)
        ctx.beta_gate = beta_gate
        ctx.beta_up = beta_up
        return output

    @staticmethod
    def backward(
        ctx,
        output_gradient: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        gate, up = ctx.saved_tensors
        output_gradient = output_gradient.contiguous()
        gate_gradient = torch.empty_like(gate)
        up_gradient = torch.empty_like(up)
        elements = gate.numel()
        block = 256
        _situ_glu_backward[(triton.cdiv(elements, block),)](
            gate,
            up,
            output_gradient,
            gate_gradient,
            up_gradient,
            elements=elements,
            beta_gate=ctx.beta_gate,
            beta_up=ctx.beta_up,
            block=block,
            num_warps=4,
        )
        return gate_gradient, up_gradient, None, None


def situ_glu(
    gate: torch.Tensor,
    up: torch.Tensor,
    beta_gate: float = 4.0,
    beta_up: float = 25.0,
) -> torch.Tensor:
    """Dispatch SiTU-GLU to one fused CUDA kernel per forward/backward pass."""

    if gate.shape != up.shape:
        raise ValueError("SiTU-GLU gate and up tensors must have matching shapes")
    if beta_gate <= 0 or beta_up <= 0:
        raise ValueError("SiTU-GLU beta values must be positive")
    if _SITU_GLU_BACKEND == "reference":
        return situ_glu_reference(gate, up, beta_gate, beta_up)
    can_use_triton = (
        triton is not None
        and gate.is_cuda
        and up.is_cuda
        and gate.dtype in {torch.float16, torch.bfloat16, torch.float32}
        and up.dtype == gate.dtype
    )
    if _SITU_GLU_BACKEND == "triton" and not can_use_triton:
        raise RuntimeError(
            "ASTER_SITU_GLU_BACKEND=triton requires matching FP16/BF16/FP32 CUDA tensors "
            "and the triton package"
        )
    if can_use_triton:
        return _TritonSiTUGLU.apply(gate, up, float(beta_gate), float(beta_up))
    return situ_glu_reference(gate, up, beta_gate, beta_up)
