from scripts.run_tracking_roundtrip import history_has_resume


def test_tracking_history_requires_both_phases_at_exact_steps() -> None:
    assert history_has_resume(
        [
            {"_step": 0, "aster_roundtrip_phase": 1},
            {"_step": 1, "aster_roundtrip_phase": 2},
        ]
    )
    assert not history_has_resume([{"_step": 0, "aster_roundtrip_phase": 1}])
    assert not history_has_resume(
        [
            {"_step": 1, "aster_roundtrip_phase": 1},
            {"_step": 0, "aster_roundtrip_phase": 2},
        ]
    )
