from __future__ import annotations

import subprocess

import pytest

from scripts.profile_training import active_compute_processes


def test_active_compute_processes_parses_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: list[list[str]] = []

    def fake_check_output(command: list[str], **_: object) -> str:
        observed.append(command)
        return "123, python, 2048, GPU-a\n456, [Not Found], [N/A], GPU-a\n"

    monkeypatch.setattr(subprocess, "check_output", fake_check_output)

    assert active_compute_processes(2) == [
        {
            "pid": 123,
            "process_name": "python",
            "used_gpu_memory_mib": "2048",
            "gpu_uuid": "GPU-a",
        },
        {
            "pid": 456,
            "process_name": "[Not Found]",
            "used_gpu_memory_mib": "[N/A]",
            "gpu_uuid": "GPU-a",
        },
    ]
    assert observed[0][-1] == "2"


def test_active_compute_processes_accepts_empty_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "check_output", lambda *_args, **_kwargs: "")
    assert active_compute_processes(0) == []


def test_active_compute_processes_fails_when_audit_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*_args: object, **_kwargs: object) -> str:
        raise subprocess.TimeoutExpired("nvidia-smi", 5)

    monkeypatch.setattr(subprocess, "check_output", fail)
    with pytest.raises(RuntimeError, match="Cannot audit"):
        active_compute_processes(0)
