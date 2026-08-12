from __future__ import annotations

import pytest
import torch

from asterlm.kernels.situ_glu import situ_glu, situ_glu_reference


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA parity gate")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_situ_glu_forward_backward_matches_oracle(dtype: torch.dtype) -> None:
    torch.manual_seed(19)
    device = torch.device("cuda")
    # Chunked gate/up tensors exercise the non-contiguous layout emitted by gate_up.
    packed = torch.randn(257, 1026, device=device, dtype=dtype)
    gate, up = packed.chunk(2, dim=-1)
    reference_gate = gate.detach().clone().requires_grad_(True)
    reference_up = up.detach().clone().requires_grad_(True)
    fused_gate = gate.detach().requires_grad_(True)
    fused_up = up.detach().requires_grad_(True)

    reference = situ_glu_reference(reference_gate, reference_up)
    fused = situ_glu(fused_gate, fused_up)
    tolerance = 2e-5 if dtype == torch.float32 else 2e-2
    torch.testing.assert_close(fused, reference, atol=tolerance, rtol=tolerance)

    gradient = torch.randn_like(reference)
    reference.backward(gradient)
    fused.backward(gradient)
    torch.testing.assert_close(
        fused_gate.grad,
        reference_gate.grad,
        atol=tolerance * 2,
        rtol=tolerance * 2,
    )
    torch.testing.assert_close(
        fused_up.grad,
        reference_up.grad,
        atol=tolerance * 2,
        rtol=tolerance * 2,
    )
