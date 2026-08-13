from __future__ import annotations

import pytest

from asterlm.cuda_allocator import DEFAULT_CUDA_ALLOC_CONF, cuda_allocator_environment


def test_cuda_allocator_defaults_are_explicit_for_old_and_new_pytorch_names():
    result = cuda_allocator_environment({"UNCHANGED": "yes"})
    assert DEFAULT_CUDA_ALLOC_CONF == "backend:native"
    assert result["PYTORCH_ALLOC_CONF"] == DEFAULT_CUDA_ALLOC_CONF
    assert result["PYTORCH_CUDA_ALLOC_CONF"] == DEFAULT_CUDA_ALLOC_CONF
    assert result["UNCHANGED"] == "yes"


def test_cuda_allocator_preserves_one_user_policy_but_rejects_conflicts():
    result = cuda_allocator_environment({"PYTORCH_ALLOC_CONF": "max_split_size_mb:128"})
    assert result["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:128"
    with pytest.raises(RuntimeError, match="disagree"):
        cuda_allocator_environment(
            {
                "PYTORCH_ALLOC_CONF": "expandable_segments:True",
                "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128",
            }
        )
