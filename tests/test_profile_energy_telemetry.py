from __future__ import annotations

from scripts.profile_training import summarize_gpu_samples


def test_gpu_power_summary_records_mean_for_total_energy_estimation():
    samples = [
        {"phase": "warmup", "power.draw": 300.0},
        {"phase": "measured", "power.draw": 90.0, "utilization.gpu": 80.0},
        {"phase": "measured", "power.draw": 110.0, "utilization.gpu": 100.0},
    ]
    summary = summarize_gpu_samples(samples)
    assert summary["measured_sample_count"] == 2
    assert summary["mean_power_draw"] == 100.0
    assert summary["median_utilization_gpu"] == 90.0
