import pytest

from scripts.import_final_scale_equal_wall_evidence import LARGE, SMALL, validate_analyses


def _analysis(seed: int, *, large_wall: float = 6.5) -> dict:
    return {
        "campaign_status": "complete",
        "analysis_status": "complete",
        "common_budgets_by_seed": {str(seed): {"wall_clock_total_seconds": 100.0}},
        "candidates": {
            SMALL: {
                "equal_wall_loss_mean": 6.7,
                "equal_wall_loss_by_seed": {str(seed): 6.7},
                "final_eval_loss_mean": 6.7,
                "final_eval_loss_by_seed": {str(seed): 6.7},
                "equal_active_flops_loss_mean": 6.7,
                "equal_active_flops_loss_by_seed": {str(seed): 6.7},
                "median_training_tokens_per_second": 3000.0,
                "median_training_tokens_per_second_by_seed": {str(seed): 3000.0},
            },
            LARGE: {
                "equal_wall_loss_mean": large_wall,
                "equal_wall_loss_by_seed": {str(seed): large_wall},
                "final_eval_loss_mean": 6.4,
                "final_eval_loss_by_seed": {str(seed): 6.4},
                "equal_active_flops_loss_mean": 6.6,
                "equal_active_flops_loss_by_seed": {str(seed): 6.6},
                "median_training_tokens_per_second": 2500.0,
                "median_training_tokens_per_second_by_seed": {str(seed): 2500.0},
            },
        },
        "runs": [
            {
                "seed": seed,
                "candidate_id": key.split(":")[0],
                "execution_variant": key.split(":")[1],
                "status": "ok",
                "tokens_seen": 1_048_576,
                "model_config_sha256": "a" if key == SMALL else "b",
            }
            for key in (SMALL, LARGE)
        ],
    }


def test_two_seed_final_scale_gate_requires_each_seed_to_win() -> None:
    result = validate_analyses([_analysis(1337), _analysis(2027, large_wall=6.6)])
    assert set(result["seeds"]) == {1337, 2027}
    assert result["mean_large_equal_wall_loss_improvement"] == pytest.approx(0.15)

    with pytest.raises(ValueError, match="did not win equal-wall"):
        validate_analyses([_analysis(1337), _analysis(2027, large_wall=6.8)])


def test_combined_two_seed_artifact_is_accepted() -> None:
    first = _analysis(1337)
    second = _analysis(2027, large_wall=6.6)
    first["runs"].extend(second["runs"])
    first["common_budgets_by_seed"].update(second["common_budgets_by_seed"])
    for key in (SMALL, LARGE):
        for metric in (
            "equal_wall_loss_by_seed",
            "final_eval_loss_by_seed",
            "equal_active_flops_loss_by_seed",
            "median_training_tokens_per_second_by_seed",
        ):
            first["candidates"][key][metric].update(second["candidates"][key][metric])

    result = validate_analyses([first])
    assert set(result["seeds"]) == {1337, 2027}
