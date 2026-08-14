from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import threading
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from asterlm.backends import execution_backend_manifest


def _run(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def static_system_manifest(device: torch.device) -> dict[str, Any]:
    capability = torch.cuda.get_device_capability(device) if device.type == "cuda" else None
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
        "execution_backend": execution_backend_manifest(device.type, capability),
        "pytorch_allocator": {
            "PYTORCH_ALLOC_CONF": os.environ.get("PYTORCH_ALLOC_CONF"),
            "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        },
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
            "compute_capability": list(capability) if capability else None,
            "multiprocessors": props.multi_processor_count,
            "bf16_supported": torch.cuda.is_bf16_supported(),
        }
    return manifest


def _visible_nvidia_device(device: torch.device) -> str:
    logical_index = int(device.index or 0)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        entries = [entry.strip() for entry in visible.split(",") if entry.strip()]
        if logical_index < len(entries):
            return entries[logical_index]
    return str(logical_index)


def _dmon_percentile(histogram: list[int], fraction: float) -> float | None:
    total = sum(histogram)
    if total <= 0:
        return None

    def value_at(rank: int) -> float:
        seen = 0
        for value, count in enumerate(histogram):
            seen += count
            if seen > rank:
                return float(value)
        return float(len(histogram) - 1)

    position = (total - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, total - 1)
    weight = position - lower
    return value_at(lower) * (1.0 - weight) + value_at(upper) * weight


class ContinuousGpuSampler:
    """Low-overhead, phase-independent GPU telemetry from one persistent dmon process."""

    def __init__(self, device: torch.device, *, energy_joules: float = 0.0) -> None:
        self.device = device
        self.energy_joules = float(energy_joules)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._process: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None
        self._samples = 0
        self._util_sum = 0.0
        self._util_histogram = [0] * 101
        self._power_sum = 0.0
        self._power_samples = 0
        self._last_power_w: float | None = None
        self._last_time: float | None = None
        self._latest: dict[str, float] = {}

    @staticmethod
    def parse_row(line: str) -> dict[str, float] | None:
        fields = line.split()
        if not fields or fields[0].startswith("#") or len(fields) < 15:
            return None

        def value(index: int) -> float | None:
            try:
                return float(fields[index])
            except (IndexError, ValueError):
                return None

        mapped = {
            "gpu_power_w": value(1),
            "gpu_temperature_c": value(2),
            "gpu_util_percent": value(4),
            "gpu_mem_util_percent": value(5),
            "gpu_mem_clock_mhz": value(10),
            "gpu_sm_clock_mhz": value(11),
            "gpu_memory_used_mib": value(12),
        }
        return {key: item for key, item in mapped.items() if item is not None}

    def start(self) -> bool:
        if self.device.type != "cuda" or shutil.which("nvidia-smi") is None:
            return False
        try:
            self._process = subprocess.Popen(
                [
                    "nvidia-smi",
                    "dmon",
                    "-s",
                    "pucm",
                    "-d",
                    "1",
                    "-i",
                    _visible_nvidia_device(self.device),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except OSError:
            self._process = None
            return False
        self._thread = threading.Thread(
            target=self._run,
            name="aster-continuous-gpu-telemetry",
            daemon=True,
        )
        self._thread.start()
        return True

    def _run(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            if self._stop.is_set():
                break
            sample = self.parse_row(line)
            if not sample:
                continue
            now = time.monotonic()
            with self._lock:
                utilization = sample.get("gpu_util_percent")
                if utilization is not None:
                    bounded = max(0.0, min(100.0, utilization))
                    self._samples += 1
                    self._util_sum += bounded
                    self._util_histogram[round(bounded)] += 1
                power = sample.get("gpu_power_w")
                if power is not None:
                    self._power_sum += power
                    self._power_samples += 1
                if self._last_time is not None and self._last_power_w is not None:
                    self.energy_joules += self._last_power_w * max(0.0, now - self._last_time)
                self._last_time = now
                if power is not None:
                    self._last_power_w = power
                self._latest = sample

    def snapshot(self) -> dict[str, float]:
        with self._lock:
            result = dict(self._latest)
            result["gpu_time_sample_count"] = float(self._samples)
            if self._samples:
                result["gpu_util_time_mean_percent"] = self._util_sum / self._samples
                for label, fraction in (("p10", 0.10), ("p50", 0.50), ("p90", 0.90)):
                    percentile = _dmon_percentile(self._util_histogram, fraction)
                    if percentile is not None:
                        result[f"gpu_util_time_{label}_percent"] = percentile
            if self._power_samples:
                result["gpu_power_time_mean_w"] = self._power_sum / self._power_samples
            result["gpu_energy_joules_total"] = self.energy_joules
            result["gpu_energy_kwh_total"] = self.energy_joules / 3_600_000.0
            return result

    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def close(self) -> None:
        self._stop.set()
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        if self._thread is not None:
            self._thread.join(timeout=3)


@dataclass
class SystemSampler:
    device: torch.device
    min_interval: float = 5.0
    continuous_gpu: bool = False
    _last_time: float = 0.0
    _last: dict[str, float] | None = None
    energy_joules: float = 0.0
    _last_energy_time: float | None = None
    _last_power_w: float | None = None
    _continuous: ContinuousGpuSampler | None = None

    def start_continuous(self) -> None:
        if self._continuous is None and self.continuous_gpu and self.device.type == "cuda":
            monitor = ContinuousGpuSampler(self.device, energy_joules=self.energy_joules)
            if monitor.start():
                self._continuous = monitor

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
            monitor = self._continuous
            continuous = monitor.snapshot() if monitor is not None else {}
            if monitor is not None and not continuous.get("gpu_time_sample_count", 0) and not monitor.running():
                monitor.close()
                self._continuous = None
                monitor = None
            if continuous.get("gpu_time_sample_count", 0) > 0:
                out.update(continuous)
                self.energy_joules = float(out.get("gpu_energy_joules_total", self.energy_joules))
            else:
                query = _run(
                    [
                        "nvidia-smi",
                        "--query-gpu=temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu,utilization.memory,memory.used",
                        "--format=csv,noheader,nounits",
                        "-i",
                        _visible_nvidia_device(self.device),
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
                    except (TypeError, ValueError):
                        # `nvidia-smi` may emit N/A for unsupported counters. Other
                        # process and CUDA metrics remain valid for this sample.
                        pass
                power = out.get("gpu_power_w")
                if monitor is None:
                    if self._last_energy_time is not None and self._last_power_w is not None:
                        self.energy_joules += self._last_power_w * max(
                            0.0, now - self._last_energy_time
                        )
                    self._last_energy_time = now
                    if power is not None:
                        self._last_power_w = power
                out["gpu_energy_joules_total"] = self.energy_joules
                out["gpu_energy_kwh_total"] = self.energy_joules / 3_600_000.0
        self._last_time = now
        self._last = out
        return dict(out)

    def close(self) -> None:
        if self._continuous is not None:
            self._continuous.close()
            self._continuous = None


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


def gradient_coverage(model: torch.nn.Module) -> dict[str, Any]:
    """Audit structural gradient coverage without synchronizing tensor values.

    A trainable recurrent/attention block with no gradients is always a broken
    execution path, not normal MoE sparsity. This specifically guards against
    reentrant activation checkpointing receiving a frozen embedding output and
    silently detaching the complete backbone.
    """

    trainable_tensors = 0
    tensors_with_grad = 0
    mixer_blocks: set[int] = set()
    mixer_blocks_with_grad: set[int] = set()
    embedding_head_tensors = 0
    embedding_head_with_grad = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        trainable_tensors += 1
        has_grad = parameter.grad is not None
        tensors_with_grad += int(has_grad)
        parts = name.split(".")
        if len(parts) >= 4 and parts[0] == "blocks" and parts[2] == "mixer":
            block = int(parts[1])
            mixer_blocks.add(block)
            if has_grad:
                mixer_blocks_with_grad.add(block)
        lower = name.lower()
        if lower.startswith(("token_embedding.", "lm_head.", "embedding_in_proj.", "embedding_out_proj.")):
            embedding_head_tensors += 1
            embedding_head_with_grad += int(has_grad)
    missing_mixer_blocks = sorted(mixer_blocks - mixer_blocks_with_grad)
    return {
        "trainable_gradient_tensor_count": tensors_with_grad,
        "trainable_tensor_count": trainable_tensors,
        "trainable_gradient_tensor_fraction": tensors_with_grad / max(trainable_tensors, 1),
        "mixer_block_count": len(mixer_blocks),
        "mixer_blocks_with_gradients": len(mixer_blocks_with_grad),
        "missing_mixer_blocks": missing_mixer_blocks,
        "embedding_head_tensor_count": embedding_head_tensors,
        "embedding_head_tensors_with_gradients": embedding_head_with_grad,
    }


def assert_required_gradient_coverage(model: torch.nn.Module) -> dict[str, Any]:
    coverage = gradient_coverage(model)
    failures: list[str] = []
    if coverage["missing_mixer_blocks"]:
        failures.append(f"mixer blocks {coverage['missing_mixer_blocks']}")
    if (
        coverage["embedding_head_tensor_count"]
        and not coverage["embedding_head_tensors_with_gradients"]
    ):
        failures.append("embedding/output head")
    if failures:
        raise RuntimeError(
            "Required trainable subsystems received no gradients: " + ", ".join(failures)
        )
    return coverage


@torch.no_grad()
def parameter_diagnostics(model: torch.nn.Module) -> dict[str, float]:
    """Low-frequency parameter health metrics grouped by subsystem."""
    categories: dict[str, dict[str, float]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        value = parameter.detach().float()
        lower = name.lower()
        if ".ffn.experts." in lower or ".ffn.routed." in lower:
            category = "experts"
        elif "router" in lower or "routing_bias" in lower:
            category = "router"
        elif "mixer" in lower and "kda" in lower:
            category = "kda"
        elif "mixer" in lower:
            category = "attention"
        elif "embedding" in lower or "lm_head" in lower:
            category = "embedding_head"
        else:
            category = "other"
        item = categories.setdefault(category, {"sum_sq": 0.0, "count": 0.0, "max_abs": 0.0})
        item["sum_sq"] += float(value.square().sum())
        item["count"] += float(value.numel())
        item["max_abs"] = max(item["max_abs"], float(value.abs().max()))

    result: dict[str, float] = {}
    total_sq = 0.0
    total_count = 0.0
    for category, item in categories.items():
        total_sq += item["sum_sq"]
        total_count += item["count"]
        result[f"param_rms_{category}"] = (item["sum_sq"] / max(item["count"], 1.0)) ** 0.5
        result[f"param_max_abs_{category}"] = item["max_abs"]
    result["param_global_l2"] = total_sq**0.5
    result["param_global_rms"] = (total_sq / max(total_count, 1.0)) ** 0.5
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
