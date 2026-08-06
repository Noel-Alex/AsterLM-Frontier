import pytest

from asterlm.optim import learning_rate_multiplier


def test_wsd_has_warmup_stable_and_decay_phases():
    assert learning_rate_multiplier(0, 10, 100) == pytest.approx(0.1)
    assert learning_rate_multiplier(20, 10, 100, decay_fraction=0.1) == pytest.approx(1.0)
    assert learning_rate_multiplier(89, 10, 100, decay_fraction=0.1) == pytest.approx(1.0)
    assert learning_rate_multiplier(99, 10, 100, min_ratio=0.1, decay_fraction=0.1) == pytest.approx(0.1)


def test_cosine_schedule_reaches_floor():
    value = learning_rate_multiplier(99, 10, 100, min_ratio=0.2, schedule_type="cosine")
    assert value == pytest.approx(0.2)
