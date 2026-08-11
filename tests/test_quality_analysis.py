from asterlm.experiments.quality_analysis import (
    first_budget_at_or_below,
    interpolate_loss,
    normalized_curve_auc,
)


def test_quality_curve_budget_metrics_interpolate_and_normalize():
    curve = [
        {"tokens_seen": 100, "wall_clock_total_seconds": 10.0, "eval_main_loss": 4.0},
        {"tokens_seen": 200, "wall_clock_total_seconds": 30.0, "eval_main_loss": 2.0},
    ]
    assert interpolate_loss(curve, "wall_clock_total_seconds", 20.0) == 3.0
    assert normalized_curve_auc(curve, "tokens_seen") == 3.0
    assert interpolate_loss(curve, "wall_clock_total_seconds", 5.0) is None


def test_first_budget_at_or_below_interpolates_first_crossing():
    curve = [
        {"tokens_seen": 10, "eval_main_loss": 5.0},
        {"tokens_seen": 20, "eval_main_loss": 4.0},
        {"tokens_seen": 30, "eval_main_loss": 4.2},
        {"tokens_seen": 40, "eval_main_loss": 3.0},
    ]
    assert first_budget_at_or_below(curve, "tokens_seen", 4.5) == 15.0
    assert first_budget_at_or_below(curve, "tokens_seen", 2.0) is None
