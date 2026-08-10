#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import site
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from importlib.metadata import distributions
from pathlib import Path
from typing import Any

from asterlm.runtime import configure_transformer_engine_runtime

configure_transformer_engine_runtime()


def command_output(command: list[str], timeout: int = 30) -> str | None:
    try:
        result = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.stdout else None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_nvcc() -> Path | None:
    for variable in ("CUDA_HOME", "CUDA_PATH", "CUDA_DIR"):
        value = os.environ.get(variable)
        if value and (candidate := Path(value) / "bin/nvcc").is_file():
            return candidate
    for package_root in site.getsitepackages():
        candidate = Path(package_root) / "nvidia/cu13/bin/nvcc"
        if candidate.is_file():
            return candidate
    resolved = command_output(["bash", "-lc", "command -v nvcc"], timeout=5)
    return Path(resolved) if resolved else None


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture the exact AsterLM runtime without secrets")
    parser.add_argument("--output", default="runs/setup/runtime-manifest-latest.json")
    parser.add_argument("--capabilities", help="Optional capability report to bind by hash")
    args = parser.parse_args()

    try:
        import torch

        torch_info: dict[str, Any] = {
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cudnn": torch.backends.cudnn.version(),
        }
        if torch.cuda.is_available():
            torch_info.update(
                {
                    "gpu": torch.cuda.get_device_name(0),
                    "compute_capability": list(torch.cuda.get_device_capability(0)),
                    "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
                }
            )
    except (ImportError, RuntimeError) as exc:
        torch_info = {"error": f"{type(exc).__name__}: {exc}"}

    packages = sorted(
        {
            distribution.metadata["Name"]: distribution.version
            for distribution in distributions()
            if distribution.metadata.get("Name")
        }.items(),
        key=lambda item: item[0].lower(),
    )
    nvcc = discover_nvcc()
    capability = Path(args.capabilities).resolve() if args.capabilities else None
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "python": {"version": sys.version, "executable": sys.executable},
        "platform": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "node": platform.node(),
            "os_release": command_output(["cat", "/etc/os-release"], timeout=5),
            "wsl_interop": bool(os.environ.get("WSL_INTEROP")),
        },
        "git": {
            "commit": command_output(["git", "rev-parse", "HEAD"], timeout=5),
            "branch": command_output(["git", "branch", "--show-current"], timeout=5),
            "status_porcelain": command_output(["git", "status", "--porcelain"], timeout=5),
        },
        "torch": torch_info,
        "cuda_toolkit": {
            "nvcc_path": str(nvcc) if nvcc else None,
            "nvcc_version": command_output([str(nvcc), "--version"], timeout=10) if nvcc else None,
            "cuda_home": os.environ.get("CUDA_HOME"),
            "nvte_cuda_include_dir": os.environ.get("NVTE_CUDA_INCLUDE_DIR"),
        },
        "nvidia_smi": command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,pstate,power.limit,memory.total,compute_cap",
                "--format=csv,noheader",
            ]
        ),
        "packages": dict(packages),
    }
    if capability:
        manifest["capability_report"] = {
            "path": str(capability),
            "exists": capability.is_file(),
            "sha256": sha256(capability) if capability.is_file() else None,
        }

    output = Path(args.output).resolve()
    atomic_write_json(output, manifest)
    print(output)


if __name__ == "__main__":
    main()
