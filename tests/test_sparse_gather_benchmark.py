from __future__ import annotations

import torch

from scripts.benchmark_sparse_gather_attention import build_causal_sparse_indices


def test_sparse_benchmark_indices_are_causal_and_offset_compressed_history() -> None:
    indices, compressed_count = build_causal_sparse_indices(
        16,
        batch=2,
        local_window=3,
        compress_ratio=4,
        compressed_topk=2,
        device=torch.device("cpu"),
    )
    assert compressed_count == 4
    assert indices.shape == (2, 16, 5)
    assert torch.equal(indices[0, 0, :3], torch.tensor([-1, -1, 0]))
    assert torch.equal(indices[0, 5, :3], torch.tensor([3, 4, 5]))
    assert torch.equal(indices[0, :3, 3:], torch.full((3, 2), -1))
    for position in range(16):
        compressed = indices[0, position, 3:]
        valid = compressed[compressed >= 0] - 16
        assert torch.all(valid < (position + 1) // 4)

