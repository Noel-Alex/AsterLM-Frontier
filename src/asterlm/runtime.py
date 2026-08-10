from __future__ import annotations

import os
import site
from pathlib import Path


def configure_transformer_engine_runtime() -> Path | None:
    """Expose CUDA toolkit headers to Transformer Engine's NVRTC path.

    Fedora/system CUDA installations normally provide ``CUDA_HOME``. Minimal WSL
    environments often receive CUDA through NVIDIA Python wheels instead, where the
    headers live under ``site-packages/nvidia/cu13/include`` and are not discovered
    by Transformer Engine automatically. Explicit user settings always win.
    """

    configured = os.environ.get("NVTE_CUDA_INCLUDE_DIR")
    if configured:
        path = Path(configured)
        return path if (path / "cuda_runtime.h").is_file() else None

    candidates: list[Path] = []
    for variable in ("CUDA_HOME", "CUDA_PATH", "CUDA_DIR"):
        value = os.environ.get(variable)
        if value:
            root = Path(value)
            candidates.extend((root / "include", root / "targets/x86_64-linux/include"))
    for package_root in site.getsitepackages():
        candidates.append(Path(package_root) / "nvidia/cu13/include")

    for include_dir in candidates:
        if not (include_dir / "cuda_runtime.h").is_file():
            continue
        os.environ["NVTE_CUDA_INCLUDE_DIR"] = str(include_dir)
        wheel_cuda_root = include_dir.parent
        if (wheel_cuda_root / "bin/nvcc").is_file():
            os.environ.setdefault("CUDA_HOME", str(wheel_cuda_root))
        return include_dir
    return None
