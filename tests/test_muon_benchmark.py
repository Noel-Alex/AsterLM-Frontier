from __future__ import annotations

import argparse

import pytest

from asterlm import AsterConfig
from scripts.benchmark_muon_optimizer import muon_topology
from scripts.benchmark_muon_per_head import parse_shape


def test_parse_muon_shape():
    assert parse_shape("6x128x768") == (6, 128, 768)
    with pytest.raises(argparse.ArgumentTypeError):
        parse_shape("6x0x768")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_shape("bad")


def test_muon_topology_counts_per_head_matrices():
    config = AsterConfig(
        vocab_size=64,
        d_model=32,
        n_layers=2,
        n_heads=2,
        head_dim=16,
        ffn_hidden=64,
        max_seq_len=16,
        kda_ratio=1,
        kda_backend="torch",
        kda_num_heads=2,
        kda_head_dim=16,
        latent_rank=8,
        rope_dim=8,
        attention_window=None,
        sink_tokens=0,
        mtp_depth=0,
        gradient_checkpointing=False,
    )
    groups, summary = muon_topology(config, per_head=True)
    assert groups
    assert summary["parameter_tensor_count"] > 0
    assert summary["orthogonalized_matrix_count"] > summary["parameter_tensor_count"]
