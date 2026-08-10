from __future__ import annotations

import os
from pathlib import Path

from asterlm.runtime import configure_transformer_engine_runtime


def test_configure_transformer_engine_runtime_discovers_cuda_wheel(
    monkeypatch, tmp_path: Path
) -> None:
    include_dir = tmp_path / "nvidia" / "cu13" / "include"
    include_dir.mkdir(parents=True)
    (include_dir / "cuda_runtime.h").write_text("// test", encoding="utf-8")
    nvcc = include_dir.parent / "bin" / "nvcc"
    nvcc.parent.mkdir()
    nvcc.write_text("", encoding="utf-8")

    for variable in ("NVTE_CUDA_INCLUDE_DIR", "CUDA_HOME", "CUDA_PATH", "CUDA_DIR"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr("asterlm.runtime.site.getsitepackages", lambda: [str(tmp_path)])

    assert configure_transformer_engine_runtime() == include_dir
    assert Path(os.environ["NVTE_CUDA_INCLUDE_DIR"]) == include_dir
    assert Path(os.environ["CUDA_HOME"]) == include_dir.parent


def test_configure_transformer_engine_runtime_preserves_explicit_setting(
    monkeypatch, tmp_path: Path
) -> None:
    explicit = tmp_path / "cuda-13.1" / "include"
    explicit.mkdir(parents=True)
    (explicit / "cuda_runtime.h").write_text("// test", encoding="utf-8")
    monkeypatch.setenv("NVTE_CUDA_INCLUDE_DIR", str(explicit))
    monkeypatch.setenv("CUDA_HOME", "/usr/local/cuda-13.1")

    assert configure_transformer_engine_runtime() == explicit
    assert os.environ["CUDA_HOME"] == "/usr/local/cuda-13.1"
