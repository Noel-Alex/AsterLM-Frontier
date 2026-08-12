from __future__ import annotations

import pytest
import torch

from asterlm.kernels.sparse_gather_attention import (
    sparse_gather_attention,
    sparse_gather_attention_reference,
)


def _inputs(device: torch.device, dtype: torch.dtype = torch.float32):
    torch.manual_seed(41)
    query = torch.randn(2, 7, 3, 8, device=device, dtype=dtype, requires_grad=True)
    kv = torch.randn(2, 11, 8, device=device, dtype=dtype, requires_grad=True)
    indices = torch.randint(0, 11, (2, 7, 5), device=device)
    indices[:, :2, 2:] = -1
    sink = torch.randn(3, device=device, dtype=dtype, requires_grad=True)
    return query, kv, indices, sink


def test_torch_sparse_gather_attention_handles_sink_mask_and_gradients() -> None:
    query, kv, indices, sink = _inputs(torch.device("cpu"))
    output = sparse_gather_attention_reference(query, kv, indices, sink)
    assert output.shape == query.shape
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    for tensor in (query, kv, sink):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
    assert kv.grad is not None and kv.grad.abs().sum() > 0
    assert sink.grad is not None and sink.grad.abs().sum() > 0


def test_dispatch_rejects_triton_on_cpu() -> None:
    query, kv, indices, sink = _inputs(torch.device("cpu"))
    with pytest.raises(RuntimeError, match="requires CUDA"):
        sparse_gather_attention(query, kv, indices, sink, backend="triton")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA parity gate")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_triton_forward_backward_matches_torch_oracle(dtype: torch.dtype) -> None:
    device = torch.device("cuda")
    reference_inputs = _inputs(device, dtype)
    triton_inputs = tuple(
        item.detach().clone().requires_grad_(item.requires_grad)
        if torch.is_tensor(item) and item.is_floating_point()
        else item.clone()
        for item in reference_inputs
    )
    reference = sparse_gather_attention(*reference_inputs, backend="torch")
    optimized = sparse_gather_attention(*triton_inputs, backend="triton")
    tolerance = 2e-4 if dtype == torch.float32 else 2e-2
    torch.testing.assert_close(optimized, reference, atol=tolerance, rtol=tolerance)

    gradient = torch.randn_like(reference)
    reference.backward(gradient)
    optimized.backward(gradient)
    for reference_tensor, optimized_tensor in zip(reference_inputs, triton_inputs, strict=True):
        if not reference_tensor.is_floating_point():
            continue
        assert reference_tensor.grad is not None and optimized_tensor.grad is not None
        torch.testing.assert_close(
            optimized_tensor.grad,
            reference_tensor.grad,
            atol=tolerance * 4,
            rtol=tolerance * 4,
        )

