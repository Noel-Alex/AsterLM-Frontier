from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def _next_power_of_two(value: int) -> int:
    return 1 if value <= 1 else 1 << (value - 1).bit_length()


def normalized_fwht(x: torch.Tensor) -> torch.Tensor:
    """Normalized Walsh-Hadamard transform over the final dimension.

    This is a portable reference implementation. CUDA/Triton fused kernels should be
    preferred for production serving, but the transform is deterministic and useful for
    quality experiments on any PyTorch device.
    """
    n = x.shape[-1]
    if n & (n - 1):
        raise ValueError("FWHT length must be a power of two")
    original_shape = x.shape
    y = x.reshape(-1, n)
    h = 1
    while h < n:
        blocks = y.reshape(-1, n // (2 * h), 2, h)
        a = blocks[:, :, 0, :]
        b = blocks[:, :, 1, :]
        y = torch.stack((a + b, a - b), dim=2).reshape(-1, n)
        h *= 2
    return y.reshape(original_shape) / math.sqrt(n)


def _pad_last(x: torch.Tensor, target: int) -> torch.Tensor:
    if x.shape[-1] == target:
        return x
    return torch.nn.functional.pad(x, (0, target - x.shape[-1]))


def _pack_int4(values: torch.Tensor) -> torch.Tensor:
    """Pack signed values in [-7, 7] into uint8 nibbles."""
    if values.dtype != torch.int8:
        values = values.to(torch.int8)
    if values.shape[-1] % 2:
        values = torch.nn.functional.pad(values, (0, 1))
    unsigned = (values.clamp(-7, 7) + 8).to(torch.uint8)
    return unsigned[..., 0::2] | (unsigned[..., 1::2] << 4)


def _unpack_int4(packed: torch.Tensor, width: int) -> torch.Tensor:
    low = (packed & 0x0F).to(torch.int16) - 8
    high = ((packed >> 4) & 0x0F).to(torch.int16) - 8
    out = torch.stack((low, high), dim=-1).flatten(-2)
    return out[..., :width].to(torch.int8)


@dataclass(slots=True)
class QuantizedTensor:
    data: torch.Tensor
    scale: torch.Tensor | None
    original_shape: tuple[int, ...]
    padded_dim: int
    group_size: int
    scheme: str
    source_dtype: torch.dtype

    @property
    def num_bytes(self) -> int:
        total = self.data.numel() * self.data.element_size()
        if self.scale is not None:
            total += self.scale.numel() * self.scale.element_size()
        return int(total)


def quantize_tensor(
    x: torch.Tensor,
    scheme: str,
    *,
    group_size: int = 64,
) -> QuantizedTensor:
    if scheme == "bfloat16":
        data = x.to(torch.bfloat16)
        return QuantizedTensor(data, None, tuple(x.shape), x.shape[-1], group_size, scheme, x.dtype)
    if scheme == "float8":
        if not hasattr(torch, "float8_e4m3fn"):
            raise RuntimeError("this PyTorch build does not expose float8_e4m3fn")
        data = x.to(torch.float8_e4m3fn)
        return QuantizedTensor(data, None, tuple(x.shape), x.shape[-1], group_size, scheme, x.dtype)
    if scheme not in {"int8", "int4", "hadamard_int4"}:
        raise ValueError(f"unsupported quantization scheme: {scheme}")
    if group_size <= 0:
        raise ValueError("group_size must be positive")

    original_shape = tuple(x.shape)
    transformed = x.float()
    padded_dim = x.shape[-1]
    if scheme == "hadamard_int4":
        padded_dim = _next_power_of_two(x.shape[-1])
        transformed = normalized_fwht(_pad_last(transformed, padded_dim))

    padded_group_dim = math.ceil(transformed.shape[-1] / group_size) * group_size
    transformed = _pad_last(transformed, padded_group_dim)
    groups = transformed.reshape(*transformed.shape[:-1], -1, group_size)
    qmax = 127.0 if scheme == "int8" else 7.0
    scale = groups.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    quant = torch.round(groups / scale).clamp(-qmax, qmax).to(torch.int8)
    flat = quant.reshape(*transformed.shape[:-1], padded_group_dim)
    data = flat if scheme == "int8" else _pack_int4(flat)
    return QuantizedTensor(
        data=data,
        scale=scale.to(torch.float16),
        original_shape=original_shape,
        padded_dim=padded_dim,
        group_size=group_size,
        scheme=scheme,
        source_dtype=x.dtype,
    )


def dequantize_tensor(q: QuantizedTensor, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    target_dtype = q.source_dtype if dtype is None else dtype
    if q.scheme in {"bfloat16", "float8"}:
        return q.data.to(target_dtype).reshape(q.original_shape)
    if q.scale is None:
        raise RuntimeError("integer quantized tensor is missing scales")
    width = q.scale.shape[-2] * q.group_size
    values = q.data if q.scheme == "int8" else _unpack_int4(q.data, width)
    groups = values.reshape(*values.shape[:-1], -1, q.group_size).float()
    restored = (groups * q.scale.float()).reshape(*values.shape[:-1], width)
    if q.scheme == "hadamard_int4":
        restored = normalized_fwht(restored[..., : q.padded_dim])
    restored = restored[..., : q.original_shape[-1]]
    return restored.to(target_dtype).reshape(q.original_shape)
