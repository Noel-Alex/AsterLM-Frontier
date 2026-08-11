from __future__ import annotations

import argparse

import pytest

from scripts.benchmark_muon_per_head import parse_shape


def test_parse_muon_shape():
    assert parse_shape("6x128x768") == (6, 128, 768)
    with pytest.raises(argparse.ArgumentTypeError):
        parse_shape("6x0x768")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_shape("bad")
