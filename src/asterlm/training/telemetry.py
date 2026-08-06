from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


def _run(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
    except Exception:
        return None


def static_system_manifest(device: torch.device) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "hostname": platform.node(),
        "cpu_count": os.cpu_count(),
        "git_commit": _run(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(_run(["git", "status", "--porcelain"])),
        "nvidia_smi": _run(["nvidia-smi", "--query-gpu=name,driver_version,pstate,power.limit,memory.total", "--format=csv,noheader"]),
    }
    try:
        import psutil

        vm = psutil.virtual_memory()
        manifest["host_ram_gib"] = vm.total / 2**30
        manifest["swap_gib"] = psutil.swap_memory().total / 2**30
    except ImportError:
        pass
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        manifest["gpu"] = {
            "name": props.name,
            "total_memory_gib": props.total_memory / 2**30,
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "multiprocessors": props.multi_processor_count,
            "bf16_supported": torch.cuda.is_bf16_supported(),
        }
    return manifest


@dataclass
class SystemSampler:
    device: torch.device
    min_interval: float = 5.0
    _last_time: float = 0.0
    _last: dict[str, float] | None = None

    def sample(self, force: bool = False) -> dict[str, float]:
        now = time.monotonic()
        if not force and self._last is not None and now - self._last_time < self.min_interval:
            return dict(self._last)
        out: dict[str, float] = {}
        try:
            import psutil

            proc = psutil.Process()
            out["host_process_rss_gb"] = proc.memory_info().rss / 2**30
            out["host_ram_used_gb"] = psutil.virtual_memory().used / 2**30
            out["host_ram_percent"] = psutil.virtual_memory().percent
            out["host_cpu_percent"] = psutil.cpu_percent(interval=None)
        except ImportError:
            pass
        if self.device.type == "cuda":
            out.update(
                {
                    "cuda_allocated_gb": torch.cuda.memory_allocated(self.device) / 2**30,
                    "cuda_reserved_gb": torch.cuda.memory_reserved(self.device) / 2**30,
                    "cuda_peak_allocated_gb": torch.cuda.max_memory_allocated(self.device) / 2**30,
                    "cuda_peak_reserved_gb": torch.cuda.max_memory_reserved(self.device) / 2**30,
                }
            )
            stats = torch.cuda.memory_stats(self.device)
            out["cuda_inactive_split_gb"] = stats.get("inactive_split_bytes.all.current", 0) / 2**30
            query = _run(
                [
                    "nvidia-smi",
                    "--query-gpu=temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu,utilization.memory,memory.used",
                    "--format=csv,noheader,nounits",
                    "-i",
                    str(self.device.index or 0),
                ]
            )
            if query:
                try:
                    vals = [float(x.strip()) for x in query.splitlines()[0].split(",")]
                    keys = [
                        "gpu_temperature_c",
                        "gpu_power_w",
                        "gpu_sm_clock_mhz",
                        "gpu_mem_clock_mhz",
                        "gpu_util_percent",
                        "gpu_mem_util_percent",
                        "gpu_memory_used_mib",
                    ]
                    out.update(dict(zip(keys, vals, strict=True)))
                except Exception:
                    pass
        self._last_time = now
        self._last = out
        return dict(out)


def gradient_diagnostics(model: torch.nn.Module) -> dict[str, Any]:
    total_sq = 0.0
    max_abs = 0.0
    finite = True
    categories: dict[str, list[float]] = {}
    for name, parameter in model.named_parameters():
        grad = parameter.grad
        if grad is None:
            continue
        g = grad.detach().float()
        total_sq += float(g.square().sum())
        max_abs = max(max_abs, float(g.abs().max()))
        finite = finite and bool(torch.isfinite(g).all())
        lower = name.lower()
        if ".ffn.experts." in lower:
            category = "experts"
        elif "router" in lower:
            category = "router"
        elif "mixer" in lower and "kda" in lower:
            category = "kda"
        elif "mixer" in lower:
            category = "attention"
        elif "embedding" in lower or "lm_head" in lower:
            category = "embedding_head"
        else:
            category = "other"
        categories.setdefault(category, []).append(float(g.square().mean().sqrt()))
    result: dict[str, Any] = {
        "grad_global_l2_unclipped": total_sq**0.5,
        "grad_max_abs": max_abs,
        "grad_all_finite": int(finite),
    }
    for category, values in categories.items():
        result[f"grad_rms_{category}"] = sum(values) / max(1, len(values))
    return result


def save_diagnostic_bundle(output_dir: str | Path, *, reason: str, extra: dict[str, Any] | None = None) -> Path:
    root = Path(output_dir)
    bundles = root / "diagnostics"
    bundles.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    staging = bundles / f"bundle-{stamp}-{reason}"
    staging.mkdir(parents=True, exist_ok=True)
    for name in ("run_manifest.json", "metrics.jsonl"):
        src = root / name
        if src.exists():
            if name == "metrics.jsonl":
                lines = src.read_text(encoding="utf-8", errors="replace").splitlines()[-2000:]
                (staging / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
            else:
                shutil.copy2(src, staging / name)
    (staging / "diagnostic.json").write_text(
        json.dumps({"reason": reason, "time_unix": time.time(), **(extra or {})}, indent=2, default=str),
        encoding="utf-8",
    )
    archive = bundles / f"{staging.name}.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in staging.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(staging))
    shutil.rmtree(staging)
    return archive
