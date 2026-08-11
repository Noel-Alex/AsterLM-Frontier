from __future__ import annotations

import pytest
import torch

from asterlm.config import AsterConfig
from asterlm.layers.compressed_sparse_attention import CompressedSparseAttentionReference
from asterlm.model import AsterLM


@pytest.mark.parametrize("compress_ratio", [4, 8])
def test_compressed_attention_is_causal_and_backpropagates(compress_ratio: int) -> None:
    torch.manual_seed(23)
    attention = CompressedSparseAttentionReference(
        input_dim=16,
        n_heads=2,
        head_dim=8,
        rope_dim=4,
        max_seq_len=32,
        local_window=4,
        compress_ratio=compress_ratio,
        index_topk=3,
        q_lora_rank=8,
        index_n_heads=2,
        index_head_dim=8,
        output_groups=1,
        output_lora_rank=8,
    )
    hidden = torch.randn(2, 16, 16, requires_grad=True)
    changed = hidden.detach().clone()
    changed[:, 11:] += 100

    output = attention(hidden)
    changed_output = attention(changed)
    assert output.shape == hidden.shape
    torch.testing.assert_close(output[:, :11], changed_output[:, :11], atol=2e-5, rtol=2e-5)
    output.square().mean().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert attention.compressor.kv_proj.weight.grad is not None


def test_csa_selected_indices_keep_local_window_and_completed_groups_only() -> None:
    torch.manual_seed(29)
    attention = CompressedSparseAttentionReference(
        input_dim=12,
        n_heads=3,
        head_dim=8,
        rope_dim=4,
        max_seq_len=32,
        local_window=3,
        compress_ratio=4,
        index_topk=2,
        q_lora_rank=6,
        index_n_heads=2,
        index_head_dim=8,
        output_lora_rank=4,
    )
    attention(torch.randn(1, 12, 12))
    indices = attention.last_selected_indices
    assert indices is not None
    assert indices.shape == (1, 12, 5)
    assert torch.equal(indices[0, 0, :3], torch.tensor([-1, -1, 0]))
    assert torch.equal(indices[0, 5, :3], torch.tensor([3, 4, 5]))
    assert torch.equal(indices[0, :3, 3:], torch.full((3, 2), -1))
    for query_position in range(12):
        compressed = indices[0, query_position, 3:]
        valid = compressed[compressed >= 0] - 12
        assert torch.all(valid < (query_position + 1) // 4)


def test_hca_attends_all_completed_compressed_groups() -> None:
    attention = CompressedSparseAttentionReference(
        input_dim=16,
        n_heads=2,
        head_dim=8,
        rope_dim=4,
        max_seq_len=32,
        local_window=2,
        compress_ratio=8,
        q_lora_rank=8,
        output_lora_rank=8,
    )
    attention(torch.randn(1, 24, 16))
    indices = attention.last_selected_indices
    assert indices is not None
    assert torch.equal(indices[0, 6, 2:], torch.full((3,), -1))
    assert torch.equal(indices[0, 7, 2:], torch.tensor([24, -1, -1]))
    assert torch.equal(indices[0, 15, 2:], torch.tensor([24, 25, -1]))
    assert torch.equal(indices[0, 23, 2:], torch.tensor([24, 25, 26]))


def test_aster_model_runs_explicit_csa_hca_pattern_and_fails_closed_for_cache() -> None:
    config = AsterConfig(
        vocab_size=64,
        d_model=16,
        n_layers=2,
        n_heads=2,
        head_dim=8,
        ffn_hidden=32,
        max_seq_len=16,
        layer_pattern=["csa", "hca"],
        rope_dim=4,
        q_lora_rank=8,
        compressed_attention_local_window=3,
        compressed_attention_hca_ratio=8,
        compressed_attention_index_topk=2,
        compressed_attention_index_n_heads=2,
        compressed_attention_index_head_dim=8,
        compressed_attention_output_lora_rank=8,
        compressed_attention_rope_theta=10000.0,
        mtp_depth=0,
        gradient_checkpointing=False,
    )
    model = AsterLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    output = model(input_ids, labels=input_ids, return_logits=False)
    assert output.loss is not None and torch.isfinite(output.loss)
    output.loss.backward()

    with pytest.raises(RuntimeError, match="cached decoding is unavailable"):
        model(input_ids[:, :1], use_cache=True)
