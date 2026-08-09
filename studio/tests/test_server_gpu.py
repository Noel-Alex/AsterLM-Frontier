from __future__ import annotations

from studio import server


def test_laptop_nvidia_smi_optional_na_field_does_not_hide_gpu(monkeypatch):
    row = (
        "NVIDIA GeForce RTX 4080 Laptop GPU, 12282, 14, 0, 40, 2.08, "
        "[N/A], 210, 405\n"
    )
    monkeypatch.setattr(
        server.subprocess,
        "check_output",
        lambda *args, **kwargs: row,
    )

    info = server.system_info()
    gpu = info["gpu"]

    assert gpu["available"] is True
    assert gpu["name"] == "NVIDIA GeForce RTX 4080 Laptop GPU"
    assert gpu["memory_total_mib"] == 12282.0
    assert gpu["power_w"] == 2.08
    assert gpu["power_limit_w"] is None
