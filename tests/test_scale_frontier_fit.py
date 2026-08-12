from scripts.run_scale_frontier_fit import build_trial_plan


def test_scale_fit_plan_keeps_update_tokens_and_descending_batches():
    trials = build_trial_plan(
        ["configs/model/demo.yaml"],
        sequences=[2048],
        batches=[1, 4, 2],
        optimizers=["adamw"],
        target_update_tokens=16_384,
        moe_implementation="cutlass",
    )
    assert [trial.micro_batch for trial in trials] == [4, 2, 1]
    assert [trial.accumulation for trial in trials] == [2, 4, 8]
    assert all(
        trial.sequence * trial.micro_batch * trial.accumulation == 16_384
        for trial in trials
    )
    assert all(trial.moe_implementation == "cutlass" for trial in trials)
