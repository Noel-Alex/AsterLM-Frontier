from __future__ import annotations

from pathlib import Path

from asterlm.cloud.execution_profiles import resolve_gpu_execution_profile

ROOT = Path(__file__).resolve().parents[1]


def test_remote_gpu_profiles_preserve_effective_batch_geometry() -> None:
    profiles = ROOT / "configs/providers/gpu_execution_profiles.yaml"
    train = ROOT / "configs/train/frontier_100b_stage1_4k.yaml"
    expected = {"L40S": (4, 8), "A100-80GB": (8, 4), "H200": (16, 2), "B300": (32, 1)}
    for gpu, geometry in expected.items():
        payload, identity = resolve_gpu_execution_profile(train, gpu, profiles_path=profiles)
        cfg = payload["train"]
        assert (cfg["micro_batch_size"], cfg["gradient_accumulation_steps"]) == geometry
        assert identity["effective_batch_tokens"] == 131072
        assert cfg["optimizer"] == "muon_adamw8bit"


def test_gcp_machine_types_resolve_to_gpu_execution_profiles() -> None:
    profiles = ROOT / "configs/providers/gpu_execution_profiles.yaml"
    train = ROOT / "configs/train/frontier_100b_stage1_4k.yaml"
    expected = {
        "g2-standard-4": ("L4", 1, 32),
        "a2-ultragpu-1g": ("A100-80GB", 8, 4),
        "a3-highgpu-1g": ("H100-80GB", 8, 4),
    }
    for machine_type, (profile_name, micro_batch, accumulation) in expected.items():
        payload, identity = resolve_gpu_execution_profile(
            train, machine_type, profiles_path=profiles
        )
        cfg = payload["train"]
        assert identity["profile"] == profile_name
        assert (cfg["micro_batch_size"], cfg["gradient_accumulation_steps"]) == (
            micro_batch,
            accumulation,
        )
        assert identity["effective_batch_tokens"] == 131072
