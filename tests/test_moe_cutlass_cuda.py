from __future__ import annotations

import importlib.util

import pytest
import torch

from asterlm.layers.moe import DeepSeekStyleMoE


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm((actual.float() - expected.float()).reshape(-1))
    denominator = torch.linalg.vector_norm(expected.float().reshape(-1)).clamp_min(1e-12)
    return float((numerator / denominator).item())


@pytest.mark.parametrize(
    "implementation", ["cutlass", "torch_grouped", "torchao_fp8", "liger"]
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA parity test")
def test_grouped_moe_matches_dropless_reference(implementation):
    if implementation != "liger" and importlib.util.find_spec("grouped_gemm") is None:
        pytest.skip("nv_grouped_gemm is not installed")
    if implementation == "torchao_fp8" and importlib.util.find_spec("torchao") is None:
        pytest.skip("torchao is not installed")
    if implementation == "torchao_fp8" and torch.cuda.get_device_capability() not in {
        (9, 0),
        (10, 0),
    }:
        pytest.skip("torchao scaled grouped MM is limited to SM90/SM100")
    if implementation == "liger" and importlib.util.find_spec("liger_kernel") is None:
        pytest.skip("liger-kernel is not installed")

    kwargs = {
        "dim": 128,
        "expert_hidden": 128,
        "num_experts": 4,
        "top_k": 2,
        "shared_experts": 0,
        "dropout": 0.0,
        "balance_strategy": "hybrid",
        "linear_backend": "torch",
    }
    torch.manual_seed(20260810)
    reference = DeepSeekStyleMoE(**kwargs, moe_impl="reference")
    candidate = DeepSeekStyleMoE(**kwargs, moe_impl=implementation)
    candidate.load_state_dict(reference.state_dict(), strict=True)
    assert candidate.state_dict().keys() == reference.state_dict().keys()

    reference = reference.cuda().to(torch.bfloat16).train()
    candidate = candidate.cuda().to(torch.bfloat16).train()
    bridge = candidate._grouped_routed
    assert bridge is not None
    pack = getattr(bridge, "pack_parameter_storage", None)
    if callable(pack):
        pack()
    x = torch.randn(4, 32, 128, device="cuda", dtype=torch.bfloat16)
    x_reference = x.detach().clone().requires_grad_(True)
    x_candidate = x.detach().clone().requires_grad_(True)

    output_reference = reference(x_reference)
    output_candidate = candidate(x_candidate)
    torch.testing.assert_close(
        output_candidate.float(),
        output_reference.float(),
        rtol=3e-2,
        atol=2e-2,
    )

    probe = torch.randn_like(output_reference)
    (output_reference.float() * probe.float()).mean().backward()
    (output_candidate.float() * probe.float()).mean().backward()
    assert _relative_l2(x_candidate.grad, x_reference.grad) < 0.04

    reference_parameters = dict(reference.named_parameters())
    candidate_parameters = dict(candidate.named_parameters())
    assert candidate_parameters.keys() == reference_parameters.keys()
    for name, expected in reference_parameters.items():
        actual = candidate_parameters[name]
        assert expected.grad is not None, name
        assert actual.grad is not None, name
        tolerance = 0.12 if implementation == "torchao_fp8" else 0.05
        assert _relative_l2(actual.grad, expected.grad) < tolerance, name

    assert bridge.cache_refreshes == 2
    candidate.zero_grad(set_to_none=True)
    candidate(x.detach())
    assert bridge.cache_refreshes == 2
    assert bridge.cache_hits == 2

    optimizer = torch.optim.SGD(candidate.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    candidate(x.detach()).float().square().mean().backward()
    optimizer.step()
    candidate(x.detach())
    assert bridge.cache_refreshes == 4
    diagnostics = getattr(bridge, "diagnostics", None)
    if callable(diagnostics):
        observed = diagnostics()
        assert observed["moe_backend_forward_calls"] == 4
        assert observed["moe_backend_weight_cache_refreshes"] == 4
        assert observed["moe_backend_weight_cache_refresh_gib"] == 0
        assert observed["moe_backend_storage_pack_count"] == 1
        assert observed["moe_backend_storage_pack_gib"] > 0
        assert observed["moe_backend_host_metadata_syncs"] == (
            4 if implementation == "cutlass" else 0
        )
