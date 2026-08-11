from __future__ import annotations

import torch

from asterlm.layers.deepseek_v4_reference import (
    GatedKVCompressor,
    MHCResidualMixer,
    compressed_sparse_topk,
    mhc_split_sinkhorn,
)


def test_gated_compressor_ratio4_matches_manual_overlap() -> None:
    compressor = GatedKVCompressor(3, 2, 4)
    with torch.no_grad():
        compressor.kv_proj.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0],
                ]
            )
        )
        compressor.gate_proj.weight.zero_()
        compressor.ape.zero_()

    hidden = torch.arange(1, 25, dtype=torch.float32).view(1, 8, 3)
    actual = compressor(hidden)
    projected = compressor.kv_proj(hidden).view(1, 2, 4, 4)
    first = projected[:, 0, :, 2:].mean(dim=1)
    second = torch.cat((projected[:, 0, :, :2], projected[:, 1, :, 2:]), dim=1).mean(dim=1)
    expected = compressor.norm(torch.stack((first, second), dim=1))
    torch.testing.assert_close(actual, expected)


def test_gated_compressor_ignores_incomplete_tail_and_backpropagates() -> None:
    torch.manual_seed(11)
    compressor = GatedKVCompressor(8, 6, 4)
    prefix = torch.randn(2, 8, 8)
    hidden = torch.cat((prefix, torch.randn(2, 3, 8)), dim=1).requires_grad_(True)
    changed_tail = hidden.detach().clone()
    changed_tail[:, 8:] += 1000

    output = compressor(hidden)
    torch.testing.assert_close(output, compressor(changed_tail))
    output.square().mean().backward()
    assert hidden.grad is not None
    assert hidden.grad[:, :8].abs().sum() > 0
    assert hidden.grad[:, 8:].abs().sum() == 0


def test_compressed_sparse_topk_is_causal_and_matches_scores() -> None:
    # One head and scalar keys make the expected ranking transparent.
    queries = torch.ones(1, 8, 1, 1)
    keys = torch.tensor([[[1.0], [4.0]]])
    weights = torch.ones(1, 8, 1)
    indices, scores = compressed_sparse_topk(
        queries,
        keys,
        weights,
        compress_ratio=4,
        topk=2,
        offset=10,
    )

    assert torch.equal(indices[0, :3], torch.full((3, 2), -1))
    assert torch.equal(indices[0, 3:7, 0], torch.full((4,), 10))
    assert torch.equal(indices[0, 3:7, 1], torch.full((4,), -1))
    assert torch.equal(indices[0, 7], torch.tensor([11, 10]))
    torch.testing.assert_close(scores[0, 7], torch.tensor([4.0, 1.0]))


def test_mhc_sinkhorn_is_nearly_doubly_stochastic_and_differentiable() -> None:
    torch.manual_seed(5)
    streams = 4
    width = (2 + streams) * streams
    mixes = torch.randn(2, 3, width, requires_grad=True)
    scales = torch.ones(3, requires_grad=True)
    base = torch.zeros(width, requires_grad=True)
    pre, post, combination = mhc_split_sinkhorn(
        mixes, scales, base, streams=streams, iterations=20
    )

    assert pre.shape == (2, 3, streams)
    assert post.shape == (2, 3, streams)
    torch.testing.assert_close(
        combination.sum(dim=-1), torch.ones(2, 3, streams), atol=2e-5, rtol=2e-5
    )
    torch.testing.assert_close(
        combination.sum(dim=-2), torch.ones(2, 3, streams), atol=2e-5, rtol=2e-5
    )
    (pre.mean() + post.mean() + combination.square().mean()).backward()
    assert mixes.grad is not None and torch.isfinite(mixes.grad).all()
    assert scales.grad is not None and torch.isfinite(scales.grad).all()
    assert base.grad is not None and torch.isfinite(base.grad).all()


def test_mhc_residual_mixer_initializes_as_stable_four_stream_reference() -> None:
    torch.manual_seed(17)
    mixer = MHCResidualMixer(8, streams=4)
    residual = torch.randn(2, 5, 4, 8, requires_grad=True)
    reduced, post, combination = mixer.reduce(residual)
    expected_reduced = residual.mean(dim=2)
    torch.testing.assert_close(reduced, expected_reduced, atol=1e-5, rtol=1e-5)

    sublayer = torch.randn(2, 5, 8, requires_grad=True)
    expanded = mixer.expand(sublayer, residual, post, combination)
    assert expanded.shape == residual.shape
    assert torch.isfinite(expanded).all()
    expanded.square().mean().backward()
    assert residual.grad is not None
    assert sublayer.grad is not None

