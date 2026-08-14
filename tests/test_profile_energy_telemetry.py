from __future__ import annotations

import torch

from asterlm.training.telemetry import (
    ContinuousGpuSampler,
    _dmon_percentile,
    static_system_manifest,
)
from scripts.profile_training import summarize_gpu_samples, summarize_torch_profiler_events


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


def test_continuous_gpu_sampler_parses_nvidia_dmon_rows_without_headers():
    assert ContinuousGpuSampler.parse_row(
        "0 94 86 - 97 49 0 0 0 0 9001 1845 10064 0 -"
    ) == {
        "gpu_power_w": 94.0,
        "gpu_temperature_c": 86.0,
        "gpu_util_percent": 97.0,
        "gpu_mem_util_percent": 49.0,
        "gpu_mem_clock_mhz": 9001.0,
        "gpu_sm_clock_mhz": 1845.0,
        "gpu_memory_used_mib": 10064.0,
    }
    assert ContinuousGpuSampler.parse_row("# gpu pwr gtemp") is None


def test_continuous_gpu_percentiles_interpolate_histogram_ranks():
    histogram = [0] * 101
    for value in (84, 89, 96, 96):
        histogram[value] += 1
    assert _dmon_percentile(histogram, 0.5) == 92.5


def test_torch_profiler_summary_is_machine_readable_and_device_time_sorted():
    class Event:
        def __init__(self, name, cpu, device, count):
            self.key = name
            self.self_cpu_time_total = cpu
            self.self_device_time_total = device
            self.count = count
            self.cpu_memory_usage = 12
            self.device_memory_usage = 34
            self.input_shapes = [[2, 8]]

    rows = summarize_torch_profiler_events(
        [Event("slow_cpu", 100, 2, 1), Event("hot_kernel", 10, 200, 7)],
        limit=1,
    )
    assert rows == [
        {
            "name": "hot_kernel",
            "count": 7,
            "self_cpu_time_us": 10.0,
            "self_device_time_us": 200.0,
            "cpu_memory_bytes": 12,
            "device_memory_bytes": 34,
            "input_shapes": "[[2, 8]]",
        }
    ]


def test_system_manifest_records_the_allocator_contract(monkeypatch):
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    allocator = static_system_manifest(torch.device("cpu"))["pytorch_allocator"]
    assert allocator == {
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
