from __future__ import annotations

import re
import site
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

BUILD_PACKAGES = (
    "nvidia-cuda-cccl",
    "nvidia-cuda-crt",
    "nvidia-cuda-nvcc",
    "nvidia-nvjitlink",
    "nvidia-nvvm",
)


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _major_minor(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    match = re.search(r"(?<!\d)(\d+)\.(\d+)", value)
    return (int(match.group(1)), int(match.group(2))) if match else None


def discover_packaged_nvcc() -> Path | None:
    for package_root in site.getsitepackages():
        candidate = Path(package_root) / "nvidia/cu13/bin/nvcc"
        if candidate.is_file():
            return candidate
    return None


def cuda_toolchain_report() -> dict[str, Any]:
    packages = {
        name: _package_version(name)
        for name in (*BUILD_PACKAGES, "nvidia-cuda-runtime")
    }
    nvcc = discover_packaged_nvcc()
    nvcc_output = None
    if nvcc is not None:
        completed = subprocess.run(
            [str(nvcc), "--version"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=10,
        )
        nvcc_output = completed.stdout.strip()
    nvcc_minor = _major_minor(packages["nvidia-cuda-nvcc"] or nvcc_output)
    runtime_minor = _major_minor(packages["nvidia-cuda-runtime"])
    mismatches: list[str] = []
    if nvcc_minor and runtime_minor and nvcc_minor != runtime_minor:
        mismatches.append(
            f"NVCC {nvcc_minor[0]}.{nvcc_minor[1]} does not match CUDA headers/runtime "
            f"{runtime_minor[0]}.{runtime_minor[1]}"
        )
    for package in BUILD_PACKAGES:
        package_minor = _major_minor(packages[package])
        if nvcc_minor and package_minor and package_minor != nvcc_minor:
            mismatches.append(
                f"{package} {packages[package]} does not match NVCC "
                f"{nvcc_minor[0]}.{nvcc_minor[1]}"
            )
    return {
        "schema_version": 1,
        "compatible": not mismatches,
        "packaged_nvcc": str(nvcc) if nvcc else None,
        "nvcc_version": nvcc_output,
        "packages": packages,
        "mismatches": mismatches,
    }


def require_compatible_cuda_toolchain() -> dict[str, Any]:
    report = cuda_toolchain_report()
    if not report["compatible"]:
        details = "; ".join(report["mismatches"])
        raise RuntimeError(
            "Incoherent pip CUDA extension-build toolchain: "
            f"{details}. For the WSL PyTorch cu130 environment run "
            "scripts/repair_cuda130_wsl.sh before compiling TileLang/CUDA kernels."
        )
    return report
