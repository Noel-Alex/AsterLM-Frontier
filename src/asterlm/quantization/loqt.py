from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .kv import QuantizedTensor, dequantize_tensor, quantize_tensor


class _LoQTLinearFunction(torch.autograd.Function):
    """INT4 base + low-rank trainable update without saving a BF16 base for backward."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        bias: torch.Tensor | None,
        qweight: torch.Tensor,
        qscale: torch.Tensor,
        in_features: int,
        out_features: int,
        group_size: int,
        adapter_scale: float,
    ) -> torch.Tensor:
        q = QuantizedTensor(
            data=qweight,
            scale=qscale,
            original_shape=(out_features, in_features),
            padded_dim=in_features,
            group_size=group_size,
            scheme="int4",
            source_dtype=x.dtype,
        )
        weight = dequantize_tensor(q, dtype=x.dtype).to(x.device)
        base = F.linear(x, weight, bias)
        low = F.linear(F.linear(x, a), b) * adapter_scale
        ctx.save_for_backward(x, a, b, qweight, qscale)
        ctx.in_features = in_features
        ctx.out_features = out_features
        ctx.group_size = group_size
        ctx.adapter_scale = adapter_scale
        ctx.has_bias = bias is not None
        return base + low

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, a, b, qweight, qscale = ctx.saved_tensors
        q = QuantizedTensor(
            data=qweight,
            scale=qscale,
            original_shape=(ctx.out_features, ctx.in_features),
            padded_dim=ctx.in_features,
            group_size=ctx.group_size,
            scheme="int4",
            source_dtype=x.dtype,
        )
        weight = dequantize_tensor(q, dtype=x.dtype).to(x.device)
        flat_x = x.reshape(-1, ctx.in_features)
        flat_grad = grad_output.reshape(-1, ctx.out_features)
        scale = ctx.adapter_scale
        grad_x = flat_grad @ weight
        grad_x = grad_x + ((flat_grad @ b) @ a) * scale
        grad_x = grad_x.reshape_as(x)
        projected = flat_x @ a.transpose(0, 1)
        grad_b = flat_grad.transpose(0, 1) @ projected * scale
        grad_a = (flat_grad @ b).transpose(0, 1) @ flat_x * scale
        grad_bias = flat_grad.sum(dim=0) if ctx.has_bias else None
        return grad_x, grad_a, grad_b, grad_bias, None, None, None, None, None, None


@dataclass(slots=True)
class LoQTMergeStats:
    modules: int = 0
    effective_weights: int = 0
    adapter_parameters: int = 0


class LoQTLinear(nn.Module):
    """VRAM-first LoQT-style linear layer for from-scratch experimentation.

    The full-rank base is stored groupwise INT4. Only BF16 low-rank A/B matrices are
    trainable. At a configured interval, the update is merged into the INT4 base on CPU
    and the adapters are reinitialized. This is intentionally called *LoQT-style*: it
    implements periodic low-rank merge/requantization, but not the paper's expensive
    gradient-SVD basis refresh. It exists to test the key VRAM trade-off on Ada hardware.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        *,
        rank: int = 32,
        alpha: float = 32.0,
        group_size: int = 64,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if rank <= 0 or group_size <= 0:
            raise ValueError("rank and group_size must be positive")
        self.in_features = in_features
        self.out_features = out_features
        self.rank = min(rank, in_features, out_features)
        self.alpha = float(alpha)
        self.group_size = group_size
        self.adapter_scale = self.alpha / self.rank
        full = torch.empty(out_features, in_features, dtype=torch.float32)
        nn.init.normal_(full, mean=0.0, std=init_std)
        quant = quantize_tensor(full, "int4", group_size=group_size)
        assert quant.scale is not None
        self.register_buffer("qweight", quant.data)
        self.register_buffer("qscale", quant.scale)
        self.a = nn.Parameter(torch.empty(self.rank, in_features))
        self.b = nn.Parameter(torch.zeros(out_features, self.rank))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.reset_adapters()
        self._aster_linear = True
        self._aster_linear_backend = "loqt_int4"
        self._is_loqt = True

    @property
    def effective_numel(self) -> int:
        return self.out_features * self.in_features + (self.out_features if self.bias is not None else 0)

    @property
    def packed_bytes(self) -> int:
        return (
            self.qweight.numel() * self.qweight.element_size()
            + self.qscale.numel() * self.qscale.element_size()
        )

    @torch.no_grad()
    def reset_adapters(self) -> None:
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.b.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _LoQTLinearFunction.apply(
            x,
            self.a,
            self.b,
            self.bias,
            self.qweight,
            self.qscale,
            self.in_features,
            self.out_features,
            self.group_size,
            self.adapter_scale,
        )

    @torch.no_grad()
    def merge_and_requantize(self, *, on_cpu: bool = True) -> None:
        target_device = self.qweight.device
        q = QuantizedTensor(
            data=self.qweight.detach().cpu() if on_cpu else self.qweight,
            scale=self.qscale.detach().cpu() if on_cpu else self.qscale,
            original_shape=(self.out_features, self.in_features),
            padded_dim=self.in_features,
            group_size=self.group_size,
            scheme="int4",
            source_dtype=torch.float32,
        )
        base = dequantize_tensor(q, dtype=torch.float32)
        a = self.a.detach().float().cpu() if on_cpu else self.a.detach().float()
        b = self.b.detach().float().cpu() if on_cpu else self.b.detach().float()
        merged = base + (b @ a) * self.adapter_scale
        new = quantize_tensor(merged, "int4", group_size=self.group_size)
        assert new.scale is not None
        self.qweight.copy_(new.data.to(target_device))
        self.qscale.copy_(new.scale.to(target_device))
        self.reset_adapters()

    @torch.no_grad()
    def scale_effective_weight(self, scale: float) -> None:
        if self.qweight.is_meta:
            return
        q = QuantizedTensor(
            data=self.qweight.detach().cpu(),
            scale=self.qscale.detach().cpu(),
            original_shape=(self.out_features, self.in_features),
            padded_dim=self.in_features,
            group_size=self.group_size,
            scheme="int4",
            source_dtype=torch.float32,
        )
        full = dequantize_tensor(q, dtype=torch.float32) * scale
        new = quantize_tensor(full, "int4", group_size=self.group_size)
        assert new.scale is not None
        self.qweight.copy_(new.data.to(self.qweight.device))
        self.qscale.copy_(new.scale.to(self.qscale.device))
        self.b.mul_(scale)
        if self.bias is not None:
            self.bias.mul_(scale)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}, "
            f"group_size={self.group_size}, bias={self.bias is not None}"
        )


def iter_loqt_modules(module: nn.Module):
    for child in module.modules():
        if isinstance(child, LoQTLinear):
            yield child


@torch.no_grad()
def merge_loqt_modules(module: nn.Module, *, on_cpu: bool = True) -> LoQTMergeStats:
    stats = LoQTMergeStats()
    for layer in iter_loqt_modules(module):
        layer.merge_and_requantize(on_cpu=on_cpu)
        stats.modules += 1
        stats.effective_weights += layer.out_features * layer.in_features
        stats.adapter_parameters += layer.a.numel() + layer.b.numel()
    return stats


def effective_parameter_count(module: nn.Module) -> int:
    loqt_parameter_ids: set[int] = set()
    total = 0
    for layer in iter_loqt_modules(module):
        total += layer.effective_numel
        loqt_parameter_ids.update(id(parameter) for parameter in layer.parameters())
    seen: set[int] = set()
    for parameter in module.parameters():
        if id(parameter) in loqt_parameter_ids or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        total += parameter.numel()
    return total
