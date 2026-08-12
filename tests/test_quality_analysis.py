from asterlm.experiments.quality_analysis import (
    first_budget_at_or_below,
    interpolate_loss,
    normalized_curve_auc,
    optimizer_screening,
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


def test_optimizer_screening_tunes_families_and_keeps_pareto_tradeoffs():
    campaign = {
        "arms": {
            "adam-fast": {"family": "adamw"},
            "adam-quality": {"family": "adamw"},
            "muon-balanced": {"family": "muon"},
            "muon-broken": {"family": "muon"},
        }
    }

    def metrics(variant, wall_loss, final_loss, throughput, **overrides):
        return {
            "execution_variant": variant,
            "complete_seed_count": 2,
            "expected_seed_count": 2,
            "run_survival_rate": 1.0,
            "training_loss_nonfinite_count": 0,
            "gradient_nonfinite_count": 0,
            "equal_wall_loss_mean": wall_loss,
            "final_eval_loss_mean": final_loss,
            "wall_curve_auc_mean": wall_loss + 0.1,
            "median_training_tokens_per_second": throughput,
            **overrides,
        }

    candidates = {
        "model:adam-fast": metrics("adam-fast", 4.1, 3.9, 1200),
        "model:adam-quality": metrics("adam-quality", 3.8, 3.7, 900),
        "model:muon-balanced": metrics("muon-balanced", 3.9, 3.8, 1100),
        "model:muon-broken": metrics(
            "muon-broken", 3.0, 3.0, 1300, gradient_nonfinite_count=1
        ),
    }
    result = optimizer_screening(campaign, candidates)
    assert result["status"] == "screen_complete"
    assert result["family_winners"]["adamw"]["execution_variant"] == "adam-quality"
    assert result["family_winners"]["muon"]["execution_variant"] == "muon-balanced"
    assert result["promotion_shortlist"] == ["adam-quality", "muon-balanced"]
    assert result["disqualified"]["model:muon-broken"] == ["nonfinite_gradient"]
    assert result["final_optimizer_selected"] is False
