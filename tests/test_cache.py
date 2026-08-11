from __future__ import annotations

import pytest

from asterlm.cache import AsterCache


def test_fla_cache_uses_transformers_compatible_adapter():
    utils = pytest.importorskip("fla.models.utils")
    cache = AsterCache.create(use_fla=True)
    assert isinstance(cache.fla_cache, utils.Cache)
    assert cache.fla_cache.get_seq_length() == 0
    cache.reset()
