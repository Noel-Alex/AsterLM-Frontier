from __future__ import annotations

import torch


def fixed_bincount(
    values: torch.Tensor,
    minlength: int,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Count bounded integer IDs without synchronizing CUDA with the host.

    ``torch.bincount`` determines its output extent from device data even when a
    minimum length is supplied. On CUDA that currently introduces a scalar
    device-to-host synchronization. MoE expert IDs and quantile-bin IDs already
    have a statically known extent, so a fixed output plus ``scatter_add_`` keeps
    the operation entirely on device.
    """

    if minlength < 1:
        raise ValueError("fixed_bincount requires minlength >= 1")
    flat = values.reshape(-1).to(dtype=torch.long)
    counts = torch.zeros(minlength, device=values.device, dtype=dtype)
    if flat.numel():
        counts.scatter_add_(0, flat, torch.ones_like(flat, dtype=dtype))
    return counts
