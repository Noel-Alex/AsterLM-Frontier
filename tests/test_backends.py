from __future__ import annotations

import pytest

from asterlm.backends import EXECUTION_BACKENDS, BackendRegistry, BackendSpec


@pytest.mark.parametrize(
    ("capability", "expected"),
    [
        ((8, 9), "cuda-ada-sm89"),
        ((9, 0), "cuda-hopper-sm90"),
        ((10, 0), "cuda-blackwell-sm100"),
        ((12, 0), "cuda-blackwell-rtx-sm120"),
        ((8, 6), "cuda-generic"),
        ((99, 0), "cuda-generic"),
    ],
)
def test_backend_resolution_is_exact_and_has_a_safe_fallback(capability, expected):
    assert EXECUTION_BACKENDS.resolve("cuda", capability).backend_id == expected


def test_cpu_uses_correctness_fallback():
    backend = EXECUTION_BACKENDS.resolve("cpu")
    assert backend.backend_id == "cpu-generic"
    assert backend.moe_candidates == ("torch_reference",)


def test_duplicate_backend_registration_is_rejected():
    registry = BackendRegistry()
    spec = BackendSpec("test", "cpu", (), "test", ("float32",), (), (), "test")
    registry.register(spec)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(spec)


def test_sm100_and_sm120_never_alias():
    sm100 = EXECUTION_BACKENDS.resolve("cuda", (10, 0))
    sm120 = EXECUTION_BACKENDS.resolve("cuda", (12, 0))
    assert sm100.backend_id != sm120.backend_id
    assert "tile_kernels" in sm100.attention_candidates
    assert "tile_kernels" not in sm120.attention_candidates
