#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import signal
import statistics
import subprocess
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path.cwd().resolve()
RUN = ROOT / "runs/frontier-vnext2"
CFG = RUN / "generated-configs"
LOG = RUN / "logs"
RES = RUN / "results"
QUALITY = RUN / "quality-runs"
GPU_LOG = RUN / "gpu_hygiene.jsonl"
RECORDS = RUN / "campaign-records"


@dataclass
class Trial:
    stage: str
    name: str
    status: str
    seconds: float
    command: list[str]
    env_delta: dict[str, str]
    result_path: str | None = None
    log_path: str | None = None
    summary: dict[str, Any] | None = None
    error: str | None = None
    gpu_before: dict[str, Any] | None = None
    gpu_after: dict[str, Any] | None = None
    cached: bool = False


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def clone_json(value: Any) -> Any:
    return json.loads(json.dumps(value))


def write_model(name: str, base: Path, changes: dict[str, Any]) -> Path:
    payload = clone_json(load_yaml(base))
    if "model" not in payload:
        payload = {"model": payload}
    payload["model"].update(changes)
    out = CFG / f"model-{name}.yaml"
    dump_yaml(out, payload)
    return out


def write_train(name: str, base: Path, changes: dict[str, Any]) -> Path:
    payload = clone_json(load_yaml(base))
    if "train" not in payload:
        payload = {"train": payload}
    payload["train"].update(changes)
    out = CFG / f"train-{name}.yaml"
    dump_yaml(out, payload)
    return out


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def run_capture(cmd: list[str], timeout: int = 30) -> str:
    try:
        p = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout, check=False)
        return p.stdout.strip()
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# GPU hygiene. The campaign parent never imports torch, so it owns no CUDA
# context. Every experiment runs in a fresh child process; when that process
# exits, its CUDA allocator/context cannot leak into the next trial.
# ---------------------------------------------------------------------------

def _parse_csv_line(line: str) -> list[str]:
    return next(csv.reader([line], skipinitialspace=True))


def gpu_snapshot() -> dict[str, Any]:
    fields = [
        "index", "name", "memory.total", "memory.used", "memory.free",
        "utilization.gpu", "temperature.gpu", "pstate", "clocks.current.sm",
        "clocks.current.memory", "power.draw",
    ]
    raw = run_capture([
        "nvidia-smi", "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"
    ])
    snap: dict[str, Any] = {"time": now(), "raw": raw}
    if raw and "NVIDIA-SMI" not in raw and "not found" not in raw.lower():
        first = raw.splitlines()[0]
        vals = _parse_csv_line(first)
        if len(vals) == len(fields):
            for k, v in zip(fields, vals, strict=True):
                v = v.strip()
                output_key = "gpu_name" if k == "name" else k
                if k in {
                    "index", "memory.total", "memory.used", "memory.free",
                    "utilization.gpu", "temperature.gpu", "clocks.current.sm",
                    "clocks.current.memory", "power.draw",
                }:
                    try:
                        snap[output_key] = float(v)
                    except Exception:
                        snap[output_key] = None
                else:
                    snap[output_key] = v

    apps_raw = run_capture([
        "nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"
    ])
    apps: list[dict[str, Any]] = []
    if apps_raw and "No running processes" not in apps_raw and "Not Supported" not in apps_raw:
        for line in apps_raw.splitlines():
            vals = _parse_csv_line(line)
            if len(vals) >= 2:
                row: dict[str, Any] = {"pid": vals[0].strip(), "process_name": vals[1].strip()}
                if len(vals) >= 3:
                    try:
                        row["used_gpu_memory_mib"] = float(vals[2].strip())
                    except Exception:
                        row["used_gpu_memory_mib"] = vals[2].strip()
                apps.append(row)
    snap["compute_apps"] = apps
    return snap


def log_gpu(event: str, snap: dict[str, Any], **extra: Any) -> None:
    GPU_LOG.parent.mkdir(parents=True, exist_ok=True)
    # Event identity wins over device metadata. The previous order let the GPU's
    # `name` field overwrite the trial name, making telemetry attribution ambiguous.
    payload = {"event": event, **snap, **extra}
    with GPU_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=str) + "\n")


def wait_for_clean_gpu(
    baseline_used: float,
    baseline_temp: float,
    *,
    label: str,
    timeout_s: int = 180,
) -> tuple[bool, dict[str, Any], str | None]:
    deadline = time.time() + timeout_s
    consecutive = 0
    last: dict[str, Any] = {}
    reason = None
    while time.time() < deadline:
        last = gpu_snapshot()
        used = last.get("memory.used")
        util = last.get("utilization.gpu")
        temp = last.get("temperature.gpu")
        apps = last.get("compute_apps") or []
        memory_ok = isinstance(used, (int, float)) and used <= baseline_used + 512.0
        util_ok = isinstance(util, (int, float)) and util <= 12.0
        temp_limit = max(72.0, baseline_temp + 12.0)
        temp_ok = not isinstance(temp, (int, float)) or temp <= temp_limit
        apps_ok = len(apps) == 0
        ok = memory_ok and util_ok and temp_ok and apps_ok
        log_gpu("idle_probe", last, label=label, ok=ok)
        if ok:
            consecutive += 1
            if consecutive >= 3:
                return True, last, None
        else:
            consecutive = 0
            problems = []
            if not memory_ok:
                problems.append(f"VRAM used={used}MiB > baseline+512={baseline_used+512:.0f}MiB")
            if not util_ok:
                problems.append(f"GPU util={util}%")
            if not temp_ok:
                problems.append(f"temperature={temp}C > {temp_limit:.0f}C")
            if not apps_ok:
                problems.append(f"external compute apps={apps}")
            reason = "; ".join(problems)
        time.sleep(1.0)
    return False, last, reason or "GPU did not reach idle criteria"


def ac_power_state() -> dict[str, Any]:
    supplies = Path("/sys/class/power_supply")
    seen = []
    online_values = []
    if supplies.is_dir():
        for d in supplies.iterdir():
            try:
                typ = (d / "type").read_text().strip().lower() if (d / "type").is_file() else ""
            except Exception:
                typ = ""
            if typ in {"mains", "usb", "usb_c", "usb-c"} or d.name.lower().startswith(("ac", "adp")):
                online = None
                try:
                    online = int((d / "online").read_text().strip())
                except Exception:
                    pass
                seen.append({"name": d.name, "type": typ, "online": online})
                if online is not None:
                    online_values.append(online)
    return {
        "supplies": seen,
        "known": bool(online_values),
        "on_ac": any(v == 1 for v in online_values) if online_values else None,
    }


# ---------------------------------------------------------------------------
# Trial execution / summaries
# ---------------------------------------------------------------------------

def summarize_profile(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not payload:
        return None
    out: dict[str, Any] = {"status": payload.get("status")}
    s = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    for key in ("median_seconds", "median_tokens_per_second"):
        if key in s:
            out[key] = s[key]
    mem = s.get("final_memory") or payload.get("memory_at_failure")
    if isinstance(mem, dict):
        out["memory"] = {k: mem.get(k) for k in (
            "allocated_gib", "reserved_gib", "peak_allocated_gib", "peak_reserved_gib", "inactive_split_gib"
        ) if k in mem}
    measured = [r for r in (payload.get("steps") or []) if not r.get("warmup")]
    tps = [float(r["tokens_per_second"]) for r in measured if isinstance(r.get("tokens_per_second"), (int, float))]
    if tps:
        out["measured_tps_median"] = statistics.median(tps)
        out["measured_tps_min"] = min(tps)
        out["measured_tps_max"] = max(tps)
        out["loss_last"] = measured[-1].get("loss")
        out["grad_norm_last"] = measured[-1].get("grad_norm")
    if "error" in payload:
        out["error"] = payload["error"]
    if "architecture" in payload:
        out["architecture"] = payload["architecture"]
    return out


def parse_metrics(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "metrics.jsonl"
    rows: list[dict[str, Any]] = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    eval_rows = [r for r in rows if isinstance(r.get("eval_main_loss"), (int, float))]
    train_rows = [r for r in rows if isinstance(r.get("main_loss"), (int, float)) and "eval_main_loss" not in r]
    tps = [float(r["tokens_per_second"]) for r in train_rows if isinstance(r.get("tokens_per_second"), (int, float))]
    mtp = [float(r["mtp_loss"]) for r in train_rows if isinstance(r.get("mtp_loss"), (int, float))]
    summary: dict[str, Any] = {
        "metrics_rows": len(rows),
        "eval_count": len(eval_rows),
        "final_eval_main_loss": float(eval_rows[-1]["eval_main_loss"]) if eval_rows else None,
        "best_eval_main_loss": min((float(r["eval_main_loss"]) for r in eval_rows), default=None),
        "final_train_main_loss": float(train_rows[-1]["main_loss"]) if train_rows else None,
        "median_tokens_per_second": statistics.median(tps) if tps else None,
        "max_tokens_seen": max((int(r.get("tokens_seen", 0)) for r in rows), default=0),
        "wall_clock_total_seconds": max((float(r.get("wall_clock_total_seconds", 0.0)) for r in rows), default=0.0),
        "mtp_loss_first": mtp[0] if mtp else None,
        "mtp_loss_last": mtp[-1] if mtp else None,
        "mtp_loss_min": min(mtp) if mtp else None,
    }
    manifest = read_json(run_dir / "run_manifest.json")
    if manifest:
        summary["architecture"] = manifest.get("architecture")
        summary["parameter_storage"] = manifest.get("parameter_storage")
        summary["optimizer_partition"] = manifest.get("optimizer_partition")
    return summary


def prune_checkpoints(run_dir: Path) -> None:
    for p in list(run_dir.rglob("*")) if run_dir.exists() else []:
        if p.is_file() and p.suffix.lower() in {".pt", ".pth", ".bin", ".safetensors"}:
            try:
                p.unlink()
            except Exception:
                pass


def record_path(stage: str, name: str) -> Path:
    return RECORDS / f"{stage}--{name}.json"


def load_cached_trial(stage: str, name: str) -> Trial | None:
    p = record_path(stage, name)
    payload = read_json(p)
    # Only reuse scientifically meaningful terminal outcomes. Transient failures
    # (busy GPU, timeout, contamination, generic error) must be retried on resume.
    if not payload or payload.get("status") not in {"ok", "oom", "skipped"}:
        return None
    try:
        payload["cached"] = True
        return Trial(**payload)
    except Exception:
        return None


def save_trial(trial: Trial) -> None:
    RECORDS.mkdir(parents=True, exist_ok=True)
    record_path(trial.stage, trial.name).write_text(json.dumps(asdict(trial), indent=2, default=str), encoding="utf-8")


def run_trial(
    stage: str,
    name: str,
    cmd: list[str],
    *,
    env_delta: dict[str, str] | None,
    baseline_used: float,
    baseline_temp: float,
    timeout_s: int,
    result_path: Path | None = None,
    use_cache: bool = True,
) -> Trial:
    if use_cache:
        cached = load_cached_trial(stage, name)
        if cached is not None:
            print(f"[{stage}] {name}: cached {cached.status}")
            return cached

    clean, before, reason = wait_for_clean_gpu(baseline_used, baseline_temp, label=f"before:{stage}:{name}")
    if not clean:
        trial = Trial(stage, name, "gpu_busy", 0.0, cmd, dict(env_delta or {}), error=reason, gpu_before=before)
        save_trial(trial)
        print(f"[{stage}] {name}: GPU not clean — {reason}")
        return trial

    env = os.environ.copy()
    delta = dict(env_delta or {})
    env.update(delta)
    LOG.mkdir(parents=True, exist_ok=True)
    log_path = LOG / f"{stage}--{name}.log"
    started = time.perf_counter()
    status = "error"
    error = None
    print(f"\n[{stage}] {name}\n$ {' '.join(cmd)}", flush=True)
    contamination: list[dict[str, Any]] = []
    proc = None
    try:
        with log_path.open("w", encoding="utf-8") as log:
            # New process group lets us reliably terminate a timed-out trial and any
            # descendants instead of leaving a CUDA-owning child behind.
            proc = subprocess.Popen(
                cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                text=True, start_new_session=True,
            )
            deadline = time.monotonic() + timeout_s
            sample_interval = max(
                0.25, float(os.environ.get("ASTER_GPU_SAMPLE_SECONDS", "0.5"))
            )
            next_probe = time.monotonic()
            while proc.poll() is None:
                if time.monotonic() >= deadline:
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(proc.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        proc.wait(timeout=10)
                    raise subprocess.TimeoutExpired(cmd, timeout_s)
                if time.monotonic() >= next_probe:
                    live = gpu_snapshot()
                    apps = live.get("compute_apps") or []
                    # The experiment itself should be the only CUDA compute process.
                    # Any other PID makes throughput/memory scientifically suspect.
                    extras = [a for a in apps if str(a.get("pid")) != str(proc.pid)]
                    if extras:
                        contamination.append({"snapshot": live, "extra_compute_apps": extras})
                    log_gpu(
                        "trial_running", live, stage=stage, name=name, child_pid=proc.pid,
                        trial_name=name, gpu_name=live.get("gpu_name"),
                        external_compute_apps=extras, power=ac_power_state(),
                        sample_interval_s=sample_interval,
                    )
                    next_probe = time.monotonic() + sample_interval
                time.sleep(0.25)
            returncode = proc.returncode

        payload = read_json(result_path) if result_path else None
        if payload and payload.get("status"):
            status = str(payload["status"])
            error = str(payload.get("error")) if payload.get("error") else None
        else:
            status = "ok" if returncode == 0 else "error"
        if returncode != 0 and status == "ok":
            status = "error"
        if returncode != 0 and not error:
            error = f"child returned {returncode}; see {log_path}"
        if contamination and status == "ok":
            status = "contaminated"
            error = (
                "unrelated CUDA compute process appeared while the trial was running; "
                "result retained for forensics but excluded from scientific selection"
            )
    except subprocess.TimeoutExpired:
        status = "timeout"
        error = f"timeout after {timeout_s}s; process group terminated"
    except Exception as exc:
        status = "error"
        error = f"{type(exc).__name__}: {exc}"
    seconds = time.perf_counter() - started

    # The child is now gone. Wait for allocator/context release and thermal recovery.
    _, after, after_reason = wait_for_clean_gpu(baseline_used, baseline_temp, label=f"after:{stage}:{name}")
    if after_reason:
        log_gpu("post_trial_not_idle", after, stage=stage, name=name, reason=after_reason)
    payload = read_json(result_path) if result_path else None
    trial = Trial(
        stage=stage, name=name, status=status, seconds=seconds, command=cmd, env_delta=delta,
        result_path=(str(result_path.relative_to(ROOT)) if result_path and result_path.exists() else None),
        log_path=str(log_path.relative_to(ROOT)), summary=summarize_profile(payload), error=error,
        gpu_before=before, gpu_after=after,
    )
    save_trial(trial)
    print(f"=> {status} in {seconds:.1f}s" + (f" | {error}" if error else ""), flush=True)
    return trial


def profile(
    stage: str,
    name: str,
    model: Path,
    train: Path,
    *,
    sequence: int,
    moe_impl: str,
    baseline_used: float,
    baseline_temp: float,
    warmup: int = 1,
    steps: int = 2,
    timeout_s: int = 3600,
    extra_env: dict[str, str] | None = None,
) -> Trial:
    out = RES / f"{stage}--{name}.json"
    env = {
        "ASTER_MOE_IMPL": moe_impl,
        "PYTORCH_ALLOC_CONF": os.environ.get("PYTORCH_ALLOC_CONF", "expandable_segments:True"),
    }
    if extra_env:
        env.update(extra_env)
    cmd = [
        sys.executable, "scripts/profile_training.py", "--model", str(model), "--train-config", str(train),
        "--sequence", str(sequence), "--steps", str(steps), "--warmup", str(warmup),
        "--optimizer", "apollo_mini", "--precision", "transformer_engine_fp8", "--json", str(out),
    ]
    return run_trial(stage, name, cmd, env_delta=env, baseline_used=baseline_used, baseline_temp=baseline_temp,
                     timeout_s=timeout_s, result_path=out)


def train_quality(
    stage: str,
    name: str,
    model: Path,
    train: Path,
    data: Path,
    *,
    moe_impl: str,
    baseline_used: float,
    baseline_temp: float,
    timeout_s: int = 10800,
) -> Trial:
    run_dir = Path(load_yaml(train)["train"]["output_dir"])
    meta_result = RES / f"{stage}--{name}.json"
    env = {
        "ASTER_MOE_IMPL": moe_impl,
        "PYTORCH_ALLOC_CONF": os.environ.get("PYTORCH_ALLOC_CONF", "expandable_segments:True"),
    }
    cmd = [sys.executable, "scripts/train_pretrain.py", "--model", str(model), "--train", str(train), "--data", str(data)]
    trial = run_trial(stage, name, cmd, env_delta=env, baseline_used=baseline_used, baseline_temp=baseline_temp,
                      timeout_s=timeout_s, result_path=None)
    summary = parse_metrics(run_dir)
    trial.summary = summary
    if trial.status == "ok" and summary.get("max_tokens_seen", 0) <= 0:
        trial.status = "error"
        trial.error = "training process returned successfully but produced no training metrics"
    meta_result.parent.mkdir(parents=True, exist_ok=True)
    meta_result.write_text(json.dumps({"trial": asdict(trial), "summary": summary}, indent=2, default=str), encoding="utf-8")
    trial.result_path = str(meta_result.relative_to(ROOT))
    save_trial(trial)
    prune_checkpoints(run_dir)
    return trial


def tps(trial: Trial) -> float:
    if not trial.summary:
        return 0.0
    return float(trial.summary.get("median_tokens_per_second") or trial.summary.get("measured_tps_median") or 0.0)


def peak(trial: Trial) -> float:
    try:
        return float((trial.summary or {}).get("memory", {}).get("peak_allocated_gib") or math.inf)
    except Exception:
        return math.inf


def eval_loss(trial: Trial) -> float:
    try:
        value = (trial.summary or {}).get("final_eval_main_loss")
        return float(value) if value is not None else math.inf
    except Exception:
        return math.inf


def make_frontier_model(name: str, *, dense: bool, backend: str, mtp_depth: int = 0) -> Path:
    base = ROOT / "configs/model/aster_moe_frontier_893m_fp8.yaml"
    changes: dict[str, Any] = {
        "attention_train_backend": backend,
        "gradient_checkpointing": True,
        "checkpoint_segment_size": 1,
        "lm_loss_backend": "torch_linear_ce",
        "linear_ce_chunking_method": "auto",
        "linear_ce_acc_policy": "compact",
        "mtp_depth": mtp_depth,
        "mtp_loss_weight": 0.0 if mtp_depth == 0 else 0.12,
    }
    if dense:
        changes.update({"ffn_type": "dense", "moe_first_dense_layers": 0})
    return write_model(name, base, changes)


def pattern(n_layers: int, ratio: int, recurrent: str = "kda") -> list[str]:
    if ratio == 0:
        return ["latent"] * n_layers
    cycle = [recurrent] * ratio + ["latent"]
    return [cycle[i % len(cycle)] for i in range(n_layers)]


def make_proxy_model(name: str, variant: str, *, backend: str) -> Path:
    base = ROOT / "configs/model/aster_220m.yaml"
    src = load_yaml(base)["model"]
    n_layers = int(src["n_layers"])
    d_model = int(src["d_model"])
    changes: dict[str, Any] = {
        "linear_backend": "transformer_engine",
        "ffn_linear_backend": None,
        "max_seq_len": 32768,
        "gradient_checkpointing": False,
        "checkpoint_segment_size": 1,
        "lm_loss_backend": "torch_linear_ce",
        "linear_ce_chunking_method": "auto",
        "linear_ce_acc_policy": "compact",
        "mtp_depth": 0,
        "mtp_loss_weight": 0.0,
        "mtp_architecture": "low_rank",
        "attention_train_backend": backend,
        "use_block_attnres": False,
        "latent_moe_post_norm": False,
    }
    if variant == "dense-kda3":
        changes.update({"ffn_type": "dense", "layer_pattern": pattern(n_layers, 3)})
    elif variant == "dense-kda1":
        changes.update({"ffn_type": "dense", "layer_pattern": pattern(n_layers, 1)})
    elif variant == "dense-kda7":
        changes.update({"ffn_type": "dense", "layer_pattern": pattern(n_layers, 7)})
    elif variant == "dense-all-kda":
        changes.update({"ffn_type": "dense", "layer_pattern": ["kda"] * n_layers})
    elif variant == "dense-all-mla":
        changes.update({"ffn_type": "dense", "layer_pattern": ["latent"] * n_layers})
    elif variant == "dense-gdn2-hybrid":
        changes.update({"ffn_type": "dense", "layer_pattern": pattern(n_layers, 3, recurrent="gdn2")})
    elif variant == "moe-grouped-kda3":
        # Shared + top2 gives three active SwiGLUs. expert_hidden=704 approximately
        # matches the active FFN FLOPs of dense hidden=2048 at d_model=768.
        changes.update({
            "ffn_type": "moe", "layer_pattern": pattern(n_layers, 3), "moe_first_dense_layers": 2,
            "moe_every": 1, "moe_num_experts": 8, "moe_top_k": 2, "moe_shared_experts": 1,
            "moe_expert_hidden": 704, "moe_balance_strategy": "hybrid",
            "moe_aux_loss_weight": 1e-4, "moe_router_z_loss_weight": 1e-5,
        })
    elif variant in {"latentmoe-kda3", "stable-latentmoe-kda3"}:
        # Active-compute matched: shared full-width expert plus two latent experts and
        # down/up projections are close to the dense FFN's active matmul count.
        changes.update({
            "ffn_type": "latent_moe", "layer_pattern": pattern(n_layers, 3),
            "latent_moe_dim": d_model // 4, "latent_moe_post_norm": variant.startswith("stable"),
            "moe_first_dense_layers": 2, "moe_every": 1, "moe_num_experts": 16,
            "moe_top_k": 2, "moe_shared_experts": 1, "moe_expert_hidden": 1280,
            "moe_balance_strategy": "hybrid", "moe_aux_loss_weight": 1e-4, "moe_router_z_loss_weight": 1e-5,
        })
    elif variant == "attnres-dense-kda3":
        changes.update({
            "ffn_type": "dense", "layer_pattern": pattern(n_layers, 3),
            "use_block_attnres": True, "attnres_block_size": 4,
        })
    else:
        raise ValueError(variant)
    return write_model(name, base, changes)


def make_frontier_variant(name: str, variant: str, *, backend: str) -> Path:
    """Scale a screened architecture choice to Aster's actual frontier width/depth.

    This is a short *scaling sanity check*, not a claim that a 220M proxy ranking
    transfers perfectly. Sparse FFN dimensions are re-solved at frontier width so
    active FFN matmul cost remains approximately matched to the dense control.
    """
    base = ROOT / "configs/model/aster_moe_frontier_893m_fp8.yaml"
    src = load_yaml(base)["model"]
    n_layers = int(src["n_layers"])
    d_model = int(src["d_model"])
    dense_hidden = int(src["ffn_hidden"])
    changes: dict[str, Any] = {
        "attention_train_backend": backend,
        "gradient_checkpointing": True,
        "checkpoint_segment_size": 1,
        "lm_loss_backend": "torch_linear_ce",
        "linear_ce_chunking_method": "auto",
        "linear_ce_acc_policy": "compact",
        "mtp_depth": 0,
        "mtp_loss_weight": 0.0,
        "mtp_architecture": "low_rank",
        "use_block_attnres": False,
        "latent_moe_post_norm": False,
    }
    if variant == "dense-kda3":
        changes.update({"ffn_type": "dense", "layer_pattern": pattern(n_layers, 3)})
    elif variant == "dense-kda1":
        changes.update({"ffn_type": "dense", "layer_pattern": pattern(n_layers, 1)})
    elif variant == "dense-kda7":
        changes.update({"ffn_type": "dense", "layer_pattern": pattern(n_layers, 7)})
    elif variant == "dense-all-kda":
        changes.update({"ffn_type": "dense", "layer_pattern": ["kda"] * n_layers})
    elif variant == "dense-all-mla":
        changes.update({"ffn_type": "dense", "layer_pattern": ["latent"] * n_layers})
    elif variant == "dense-gdn2-hybrid":
        changes.update({"ffn_type": "dense", "layer_pattern": pattern(n_layers, 3, recurrent="gdn2")})
    elif variant == "moe-grouped-kda3":
        # shared + top2 = 3 active experts, so H_expert ~= H_dense/3.
        expert_hidden = int(round((dense_hidden / 3) / 16) * 16)
        changes.update({
            "ffn_type": "moe", "layer_pattern": pattern(n_layers, 3),
            "moe_first_dense_layers": 2, "moe_every": 1, "moe_num_experts": 8,
            "moe_top_k": 2, "moe_shared_experts": 1, "moe_expert_hidden": expert_hidden,
            "moe_balance_strategy": "hybrid", "moe_aux_loss_weight": 1e-4,
            "moe_router_z_loss_weight": 1e-5,
        })
    elif variant in {"latentmoe-kda3", "stable-latentmoe-kda3"}:
        latent_dim = d_model // 4
        # Solve active FFN matmul count approximately:
        # shared(3*d*h) + top2 routed(6*l*h) + down/up(2*d*l) ~= 3*d*H_dense.
        target = 3 * d_model * dense_hidden - 2 * d_model * latent_dim
        coeff = 3 * d_model + 6 * latent_dim
        expert_hidden = max(16, int(round((target / coeff) / 16) * 16))
        changes.update({
            "ffn_type": "latent_moe", "layer_pattern": pattern(n_layers, 3),
            "latent_moe_dim": latent_dim,
            "latent_moe_post_norm": variant.startswith("stable"),
            "moe_first_dense_layers": 2, "moe_every": 1, "moe_num_experts": 16,
            "moe_top_k": 2, "moe_shared_experts": 1, "moe_expert_hidden": expert_hidden,
            "moe_balance_strategy": "hybrid", "moe_aux_loss_weight": 1e-4,
            "moe_router_z_loss_weight": 1e-5,
        })
    elif variant == "attnres-dense-kda3":
        changes.update({
            "ffn_type": "dense", "layer_pattern": pattern(n_layers, 3),
            "use_block_attnres": True, "attnres_block_size": 4,
        })
    else:
        raise ValueError(variant)
    return write_model(name, base, changes)


def make_quality_train(
    name: str,
    *,
    output_dir: Path,
    max_tokens: int,
    adam_lr: float,
    tokenizer: Path,
    optimizer: str = "apollo_mini",
    seed: int = 1337,
    muon_lr: float = 0.01,
) -> Path:
    base = ROOT / "configs/train/probe_memory_matrix.yaml"
    tokens_per_step = 2048 * 1 * 16
    steps = math.ceil(max_tokens / tokens_per_step)
    eval_interval = max(16, min(32, steps // 2 if steps >= 32 else max(1, steps // 2)))
    warmup = max(1, min(16, steps // 10))
    return write_train(name, base, {
        "output_dir": str(output_dir), "seed": seed, "device": "cuda", "dtype": "bfloat16",
        "matmul_precision": "high", "compile": False, "precision_backend": "transformer_engine_fp8",
        "fp8_recipe": "delayed", "fp8_format": "hybrid", "fp8_amax_history_len": 16,
        "activation_offload": False, "sequence_length": 2048, "micro_batch_size": 1,
        "gradient_accumulation_steps": 16, "max_steps": steps + 2, "max_tokens": max_tokens,
        "optimizer": optimizer, "adam_lr": adam_lr, "muon_lr": muon_lr,
        "warmup_steps": warmup, "schedule_type": "constant", "weight_decay": 0.1,
        "max_grad_norm": 1.0, "qk_clip_interval": 100, "log_interval": 4,
        "eval_interval": eval_interval, "eval_batches": 8, "save_interval": steps + 1000,
        "keep_last_checkpoints": 1, "tokenizer_path": str(tokenizer), "num_workers": 0,
        "pin_memory": True, "prefetch_factor": None, "tensorboard": False, "jsonl_metrics": True,
        "diagnostic_interval": max(32, steps), "save_diagnostic_bundle": False,
        "wandb_project": None, "hub_repo_id": None,
    })


def with_mtp(name: str, model_path: Path, *, architecture: str, weight: float) -> Path:
    payload = load_yaml(model_path)
    model = payload["model"]
    model["mtp_depth"] = 0 if architecture == "none" else 1
    model["mtp_loss_weight"] = 0.0 if architecture == "none" else weight
    model["mtp_architecture"] = "low_rank" if architecture in {"none", "low_rank"} else "deepseek"
    if architecture == "deepseek":
        patt = model.get("layer_pattern") or pattern(int(model["n_layers"]), int(model.get("kda_ratio", 3)))
        model["mtp_block_kind"] = patt[-1]
    out = CFG / f"model-{name}.yaml"
    dump_yaml(out, payload)
    return out


def capture_environment(baseline: dict[str, Any], power: dict[str, Any]) -> dict[str, Any]:
    package_probe = run_capture([
        sys.executable, "-c",
        "import json,torch; out={'torch':torch.__version__};\n"
        "mods=['transformer_engine','fla','triton','torchao','apollo_torch'];\n"
        "import importlib\n"
        "for m in mods:\n"
        " try:\n  x=importlib.import_module(m); out[m]=getattr(x,'__version__','unknown')\n"
        " except Exception as e: out[m]='ERROR:'+str(e)\n"
        "print(json.dumps(out))",
    ], timeout=60)
    return {
        "generated_at": now(), "python": sys.version, "git_head": run_capture(["git", "rev-parse", "HEAD"]),
        "git_status_short": run_capture(["git", "status", "--short"], timeout=60),
        "nvidia_smi": run_capture(["nvidia-smi"], timeout=30), "gpu_baseline": baseline,
        "power": power, "packages_raw": package_probe,
        "env": {k: os.environ.get(k) for k in (
            "LD_LIBRARY_PATH", "PYTORCH_ALLOC_CONF", "FLA_TRIL_PRECISION", "FLA_USE_FAST_OPS", "CUDA_PATH"
        )},
    }


def architecture_pareto(rows: list[Trial]) -> list[str]:
    good = [r for r in rows if r.status == "ok" and math.isfinite(eval_loss(r)) and tps(r) > 0]
    pareto = []
    for a in good:
        dominated = False
        for b in good:
            if a is b:
                continue
            if eval_loss(b) <= eval_loss(a) and tps(b) >= tps(a) and (eval_loss(b) < eval_loss(a) or tps(b) > tps(a)):
                dominated = True
                break
        if not dominated:
            pareto.append(a.name)
    return pareto


def build_report(summary: dict[str, Any]) -> str:
    lines = [
        "# AsterLM Frontier vNext2 Research Report", "",
        "## Research rule", "",
        "The final architecture is selected by measurements. Dense, standard MoE, LatentMoE, recurrent/global mixer ratios, MTP, and long-context mechanisms are hypotheses, not commitments.", "",
        "## GPU hygiene", "",
        "Every trial ran in a fresh child Python process. The parent campaign owns no CUDA context, waits for VRAM/utilization/temperature to return to an idle baseline, and refuses a trial when unrelated compute processes are present. This prevents a previous PyTorch caching allocator from contaminating the next experiment and surfaces external-GPU interference instead of silently accepting it.", "",
    ]
    hygiene = summary.get("gpu_hygiene", {})
    lines += [f"- Baseline used VRAM: {hygiene.get('baseline_used_mib')} MiB", f"- Baseline temperature: {hygiene.get('baseline_temp_c')} C", f"- AC power: {hygiene.get('power')}", ""]

    absorbed = summary.get("absorbed_mla", {})
    lines += ["## Exact absorbed MLA", "", "Aster's dense MLA training path and the algebraically absorbed MQA/GQA path were compared with repeated A/B/B/A runs. The absorbed implementation has separate CPU forward/gradient parity tests before CUDA performance is considered.", "", f"```json\n{json.dumps(absorbed, indent=2)}\n```", ""]

    context = summary.get("context_ladder", [])
    lines += ["## Context feasibility", "", "| trial | status | tok/s | peak GiB |", "|---|---:|---:|---:|"]
    for r in context:
        tr = Trial(**r) if isinstance(r, dict) and "stage" in r else None
        if tr:
            lines.append(f"| {tr.name} | {tr.status} | {tps(tr):.1f} | {peak(tr):.2f} |")
    lines.append("")

    sparse = summary.get("sparse_scout")
    lines += ["## Sparse-attention scout", "", "The old FlexAttention local/landmark prototype is retired and is not used as evidence against sparse attention. vNext2 only performs a pinned upstream Native Sparse Attention kernel scout. The current Aster frontier has 18 query heads and one absorbed latent KV head; the upstream selected NSA kernel requires a GQA group that is a multiple of 16, so the scout is explicitly a 16-head surrogate rather than a fake direct integration.", ""]
    if sparse:
        lines += [f"```json\n{json.dumps(sparse, indent=2)}\n```", ""]

    arch = summary.get("architecture_screen", [])
    lines += ["## Real-data architecture screen", "", "MTP is disabled in this stage. Candidates receive the same hash-disjoint proxy corpus, proxy tokenizer, token budget, seed, optimizer and initial LR. Sparse FFNs are sized approximately by *active* FFN compute rather than merely total parameter count.", "", "| candidate | status | final eval main loss | median tok/s | tokens |", "|---|---:|---:|---:|---:|"]
    for row in arch:
        tr = Trial(**row)
        lines.append(f"| {tr.name} | {tr.status} | {eval_loss(tr):.5f} | {tps(tr):.1f} | {(tr.summary or {}).get('max_tokens_seen')} |")
    lines += ["", f"Pareto candidates: {summary.get('architecture_pareto', [])}", ""]

    finals = summary.get("finalist_lr_sweep", [])
    lines += ["## Finalist LR sweep", "", "A short architecture screen is not allowed to become a one-LR verdict. Finalists are re-run from scratch across a small LR sweep at a larger token budget.", "", "| run | eval main loss | tok/s |", "|---|---:|---:|"]
    for row in finals:
        tr = Trial(**row)
        lines.append(f"| {tr.name} | {eval_loss(tr):.5f} | {tps(tr):.1f} |")
    lines += ["", f"Selected architecture/LR for the next ablation: {summary.get('selected_finalist')}", ""]

    transfer_rows = summary.get("frontier_transfer", [])
    lines += ["## Frontier-scale transfer check", "", "The proxy winner is not assumed to transfer perfectly to ~1B. Confirmed proxy finalists are rebuilt at frontier width/depth with sparse FFN active compute re-matched, then given a short real-text sanity run.", "", "| variant | status | eval main loss | tok/s |", "|---|---:|---:|---:|"]
    for row in transfer_rows:
        tr = Trial(**row)
        lines.append(f"| {tr.name} | {tr.status} | {eval_loss(tr):.5f} | {tps(tr):.1f} |")
    lines.append("")

    opt_rows = summary.get("optimizer_quality_scout", [])
    lines += ["## Optimizer quality/systems scout", "", "Muon is not rejected merely because the current implementation is slower. The selected proxy architecture receives a small real-text quality scout so sample efficiency and wall-clock throughput can be considered together.", "", "| variant | eval main loss | tok/s |", "|---|---:|---:|"]
    for row in opt_rows:
        tr = Trial(**row)
        lines.append(f"| {tr.name} | {eval_loss(tr):.5f} | {tps(tr):.1f} |")
    lines.append("")

    mtp_rows = summary.get("mtp_ablation", [])
    lines += ["## MTP", "", "MTP is ranked by **main-model validation loss**, never by total loss. At random initialization an auxiliary future-token CE near ln(vocab) is expected; the systems profiler's random labels cannot tell us whether MTP is useful. Here MTP gets real contiguous text and enough updates for its auxiliary loss trajectory to be interpretable.", "", "| variant | eval main loss | tok/s | first MTP loss | last MTP loss |", "|---|---:|---:|---:|---:|"]
    for row in mtp_rows:
        tr = Trial(**row)
        sm = tr.summary or {}
        lines.append(f"| {tr.name} | {eval_loss(tr):.5f} | {tps(tr):.1f} | {sm.get('mtp_loss_first')} | {sm.get('mtp_loss_last')} |")
    lines += ["", "The DeepSeek-style one-step MTP backend uses the actual future-token embedding plus the source hidden state, then a full Aster block and shared LM head. It deliberately does not claim a speculative decoding speedup until an incremental drafter/cache path exists.", ""]

    lines += ["## Long-context research direction", "", "For 256K training / 1M inference, the next sparse-attention step should be hardware-algorithm co-design, not a per-layer token indexer bolted onto the model. LongCat Sparse Attention shows why: scattered KV access and O(L²) aggregate indexer scoring can erase theoretical sparsity gains. Candidate future work is hierarchical/coarse-to-fine retrieval, contiguous sink/window budget, cross-layer index reuse with distillation, and an absorbed latent-KV execution path. These should be added only after the backbone tournament has a winner and with dedicated kernel/quality controls.", ""]
    return "\n".join(lines)


def package_results() -> Path:
    out = RUN / "frontier-vnext2-results.zip"
    include_roots = [CFG, LOG, RES, RECORDS]
    fixed = [
        RUN / "campaign_summary.json", RUN / "FRONTIER_VNEXT2_REPORT.md", RUN / "environment.json",
        RUN / "install_state.json", GPU_LOG, RUN / "proxy-data/manifest.json",
        RUN / "proxy-data/source-discovery.json", RUN / "proxy-data/prepare_result.json",
        RUN / "proxy-data/data-proxy.yaml",
    ]
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        seen: set[Path] = set()
        for p in fixed:
            if p.is_file() and p not in seen:
                z.write(p, p.relative_to(RUN)); seen.add(p)
        for root in include_roots:
            if not root.exists():
                continue
            for p in root.rglob("*"):
                if not p.is_file() or p in seen:
                    continue
                z.write(p, p.relative_to(RUN)); seen.add(p)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="AsterLM Frontier vNext2 adaptive research campaign")
    parser.add_argument("--mode", choices=["quick", "thorough"], default="thorough")
    parser.add_argument("--allow-battery", action="store_true")
    args = parser.parse_args()
    for d in (RUN, CFG, LOG, RES, QUALITY, RECORDS):
        d.mkdir(parents=True, exist_ok=True)

    power = ac_power_state()
    if power["known"] and power["on_ac"] is False and not (args.allow_battery or os.environ.get("ASTER_ALLOW_BATTERY") == "1"):
        raise SystemExit("Laptop appears to be on battery. Plug it in, then rerun. Set ASTER_ALLOW_BATTERY=1 only if you intentionally accept polluted performance data.")

    baseline_samples = []
    for _ in range(5):
        baseline_samples.append(gpu_snapshot()); time.sleep(1.0)
    if any(s.get("compute_apps") for s in baseline_samples):
        raise SystemExit(f"External CUDA compute process detected before campaign: {[s.get('compute_apps') for s in baseline_samples]}. Close it and rerun; vNext2 will not kill processes automatically.")
    used_values = [float(s["memory.used"]) for s in baseline_samples if isinstance(s.get("memory.used"), (int, float))]
    temp_values = [float(s["temperature.gpu"]) for s in baseline_samples if isinstance(s.get("temperature.gpu"), (int, float))]
    if not used_values:
        raise SystemExit("Could not read GPU memory from nvidia-smi; refusing performance experiments without hygiene telemetry")
    baseline_used = statistics.median(used_values)
    baseline_temp = statistics.median(temp_values) if temp_values else 50.0
    for s in baseline_samples:
        log_gpu("campaign_baseline", s)

    environment = capture_environment(baseline_samples[-1], power)
    (RUN / "environment.json").write_text(json.dumps(environment, indent=2, default=str), encoding="utf-8")

    trials: list[Trial] = []
    summary: dict[str, Any] = {
        "generated_at": now(), "mode": args.mode,
        "gpu_hygiene": {"baseline_used_mib": baseline_used, "baseline_temp_c": baseline_temp, "power": power},
        "research_rule": "measurements choose the final architecture; no dense/MoE/MTP/sparse-attention loyalty",
        "previous_vnext": read_json(ROOT / "runs/frontier-vnext/campaign_summary.json"),
    }

    # Core semantic tests first. This child may be CPU-only for the exact MLA parity test.
    core = run_trial(
        "validation", "core-tests",
        [sys.executable, "-m", "pytest", "-q", "tests/test_vnext2_core.py", "tests/test_vnext_core.py", "tests/test_model.py"],
        env_delta={}, baseline_used=baseline_used, baseline_temp=baseline_temp, timeout_s=900,
    )
    trials.append(core)
    if core.status != "ok":
        summary["fatal"] = "core semantic tests failed"
        (RUN / "campaign_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        raise SystemExit("Core vNext2 tests failed; see logs. No expensive campaign was started.")

    # Prepare a controlled real-text proxy dataset once. This is deliberately not
    # the final tokenizer/corpus; it exists to make architecture comparisons honest.
    prep_result = ROOT / "runs/frontier-vnext2/proxy-data/prepare_result.json"
    prep = run_trial(
        "data", "prepare-proxy",
        [sys.executable, "scripts/frontier_vnext2_prepare_proxy.py",
         "--train-tokens", "24000000" if args.mode == "thorough" else "8000000",
         "--val-tokens", "2400000" if args.mode == "thorough" else "800000"],
        env_delta={}, baseline_used=baseline_used, baseline_temp=baseline_temp, timeout_s=7200,
        result_path=prep_result,
    )
    trials.append(prep)
    prep_payload = read_json(prep_result) or {}
    data_path = Path(prep_payload.get("data_config", "")) if prep.status == "ok" else Path()
    tokenizer = Path(prep_payload.get("tokenizer", "")) if prep.status == "ok" else Path()

    # ----- Exact absorbed MLA A/B/B/A -----
    base_train = ROOT / "configs/train/probe_memory_matrix.yaml"
    systems_train = write_train("vnext2-systems", base_train, {
        "precision_backend": "transformer_engine_fp8", "fp8_recipe": "delayed", "fp8_amax_history_len": 16,
        "activation_offload": False, "optimizer": "apollo_mini", "gradient_accumulation_steps": 1,
    })
    model_a = make_frontier_model("frontier-reconstructed-mtp0", dense=True, backend="sdpa", mtp_depth=0)
    model_b = make_frontier_model("frontier-absorbed-mtp0", dense=True, backend="absorbed_sdpa", mtp_depth=0)
    ab_runs = []
    for label, model in (("A1-reconstructed", model_a), ("B1-absorbed", model_b), ("B2-absorbed", model_b), ("A2-reconstructed", model_a)):
        tr = profile("absorbed-mla", label, model, systems_train, sequence=4096, moe_impl="reference",
                     baseline_used=baseline_used, baseline_temp=baseline_temp, warmup=1, steps=2, timeout_s=3600)
        trials.append(tr); ab_runs.append(tr)
    def aggregate(prefix: str) -> dict[str, Any]:
        xs = [x for x in ab_runs if x.name.startswith(prefix) and x.status == "ok"]
        return {
            "runs": len(xs), "tps_median": statistics.median([tps(x) for x in xs]) if xs else None,
            "peak_gib_median": statistics.median([peak(x) for x in xs]) if xs else None,
        }
    summary["absorbed_mla"] = {"reconstructed": aggregate("A"), "absorbed": aggregate("B")}
    recon = summary["absorbed_mla"]["reconstructed"]
    absorbed = summary["absorbed_mla"]["absorbed"]
    # Both paths are algebraically parity-tested. For short/medium-context quality
    # experiments prefer measured throughput unless the absorbed path buys meaningful
    # peak-memory headroom at modest speed cost. Long-context feasibility still uses
    # absorbed MLA explicitly because avoiding reconstructed per-head K/V is its point.
    if not absorbed["runs"]:
        chosen_backend = "sdpa"
    elif not recon["runs"]:
        chosen_backend = "absorbed_sdpa"
    else:
        recon_tps = float(recon["tps_median"] or 0.0)
        abs_tps = float(absorbed["tps_median"] or 0.0)
        recon_peak = float(recon["peak_gib_median"] or math.inf)
        abs_peak = float(absorbed["peak_gib_median"] or math.inf)
        meaningful_memory_win = abs_peak <= recon_peak - 0.35
        acceptable_speed = abs_tps >= 0.90 * recon_tps
        chosen_backend = "absorbed_sdpa" if (abs_tps >= recon_tps or (meaningful_memory_win and acceptable_speed)) else "sdpa"
    summary["preferred_exact_attention_backend_for_quality_stages"] = chosen_backend
    summary["long_context_attention_backend"] = "absorbed_sdpa"

    # ----- Context ladders after the checkpoint bug fix, MTP off -----
    dense_context = make_frontier_model("frontier-dense-context", dense=True, backend="absorbed_sdpa", mtp_depth=0)
    context_rows: list[Trial] = []
    for seq in (8192, 16384, 32768, 65536):
        tr = profile("context", f"dense-absorbed-mtp0-{seq}", dense_context, systems_train, sequence=seq,
                     moe_impl="reference", baseline_used=baseline_used, baseline_temp=baseline_temp,
                     warmup=1 if seq <= 32768 else 0, steps=2 if seq <= 32768 else 1,
                     timeout_s=7200 if seq >= 32768 else 3600)
        trials.append(tr); context_rows.append(tr)
        if tr.status in {"oom", "error", "timeout", "gpu_busy"}:
            break
    # Conditionally test 128K only when 64K leaves a genuinely large safety margin.
    if context_rows and context_rows[-1].name.endswith("65536") and context_rows[-1].status == "ok" and peak(context_rows[-1]) <= 7.5:
        tr = profile("context", "dense-absorbed-mtp0-131072", dense_context, systems_train, sequence=131072,
                     moe_impl="reference", baseline_used=baseline_used, baseline_temp=baseline_temp,
                     warmup=0, steps=1, timeout_s=10800)
        trials.append(tr); context_rows.append(tr)
    summary["context_ladder"] = [asdict(x) for x in context_rows]

    moe_context = make_frontier_model("frontier-moe-context", dense=False, backend="absorbed_sdpa", mtp_depth=0)
    moe_rows = []
    for seq in (4096, 8192, 16384):
        tr = profile("context-moe", f"grouped-absorbed-mtp0-{seq}", moe_context, systems_train, sequence=seq,
                     moe_impl="grouped", baseline_used=baseline_used, baseline_temp=baseline_temp,
                     warmup=1, steps=2, timeout_s=5400)
        trials.append(tr); moe_rows.append(tr)
        if tr.status != "ok" or peak(tr) >= 10.5:
            break
    summary["moe_context_ladder"] = [asdict(x) for x in moe_rows]

    # ----- Pinned upstream sparse-kernel potential scout. -----
    sparse_out = RES / "sparse-scout.json"
    sparse_trial = run_trial(
        "sparse", "native-nsa-surrogate",
        [sys.executable, "scripts/frontier_vnext2_sparse_scout.py", "--output", str(sparse_out), "--repeats", "2"],
        env_delta={"PYTORCH_ALLOC_CONF": "expandable_segments:True"}, baseline_used=baseline_used,
        baseline_temp=baseline_temp, timeout_s=5400, result_path=sparse_out,
    )
    trials.append(sparse_trial)
    summary["sparse_scout"] = read_json(sparse_out)

    # ----- Real-data architecture tournament. -----
    arch_rows: list[Trial] = []
    model_map: dict[str, Path] = {}
    if prep.status == "ok" and data_path.is_file() and tokenizer.is_file():
        variants = [
            "dense-kda3", "dense-kda1", "dense-kda7", "dense-all-kda", "dense-all-mla",
            "dense-gdn2-hybrid", "moe-grouped-kda3", "latentmoe-kda3",
            "stable-latentmoe-kda3", "attnres-dense-kda3",
        ]
        order = list(variants)
        random.Random(1337).shuffle(order)
        screen_tokens = 4_194_304 if args.mode == "thorough" else 1_048_576
        for variant in order:
            model = make_proxy_model(f"screen-{variant}", variant, backend=chosen_backend)
            model_map[variant] = model
            train = make_quality_train(
                f"screen-{variant}", output_dir=QUALITY / "screen" / variant,
                max_tokens=screen_tokens, adam_lr=3e-4, tokenizer=tokenizer,
            )
            impl = "grouped" if variant in {"moe-grouped-kda3", "latentmoe-kda3", "stable-latentmoe-kda3"} else "reference"
            tr = train_quality("arch-screen", variant, model, train, data_path, moe_impl=impl,
                               baseline_used=baseline_used, baseline_temp=baseline_temp,
                               timeout_s=14400 if args.mode == "thorough" else 7200)
            trials.append(tr); arch_rows.append(tr)
        # Repeat the dense control at the end to quantify long-run thermal/data drift.
        ctrl_model = model_map.get("dense-kda3") or make_proxy_model("screen-dense-kda3", "dense-kda3", backend=chosen_backend)
        ctrl_train = make_quality_train(
            "screen-dense-kda3-repeat", output_dir=QUALITY / "screen" / "dense-kda3-repeat",
            max_tokens=screen_tokens, adam_lr=3e-4, tokenizer=tokenizer,
        )
        repeat = train_quality("arch-screen", "dense-kda3-repeat", ctrl_model, ctrl_train, data_path,
                               moe_impl="reference", baseline_used=baseline_used, baseline_temp=baseline_temp,
                               timeout_s=14400 if args.mode == "thorough" else 7200)
        trials.append(repeat); arch_rows.append(repeat)
    summary["architecture_screen"] = [asdict(x) for x in arch_rows]
    ranking_rows = [x for x in arch_rows if x.name != "dense-kda3-repeat" and x.status == "ok" and math.isfinite(eval_loss(x))]
    ranking_rows.sort(key=lambda x: (eval_loss(x), -tps(x)))
    summary["architecture_ranking"] = [{"name": x.name, "eval_main_loss": eval_loss(x), "tps": tps(x)} for x in ranking_rows]
    summary["architecture_pareto"] = architecture_pareto([x for x in arch_rows if x.name != "dense-kda3-repeat"])

    # ----- Finalist LR sweep: control + two best alternatives. -----
    finalist_rows: list[Trial] = []
    finalists: list[str] = []
    if ranking_rows and prep.status == "ok":
        # Always keep the dense KDA3 control when it is healthy. Then retain the
        # best-quality alternative and the fastest non-dominated alternative. This
        # prevents a slightly worse-but-much-faster architecture from being discarded
        # by a tiny equal-token screen.
        if any(x.name == "dense-kda3" for x in ranking_rows):
            finalists.append("dense-kda3")
        quality_alt = next((x.name for x in ranking_rows if x.name not in finalists), None)
        if quality_alt is not None:
            finalists.append(quality_alt)
        pareto_names = set(summary.get("architecture_pareto", []))
        speed_alt_rows = [
            x for x in ranking_rows if x.name not in finalists and x.name in pareto_names
        ]
        if not speed_alt_rows:
            speed_alt_rows = [x for x in ranking_rows if x.name not in finalists]
        if speed_alt_rows:
            finalists.append(max(speed_alt_rows, key=tps).name)
        finalists = finalists[:3]
        sweep_tokens = 8_388_608 if args.mode == "thorough" else 2_097_152
        lrs = [2e-4, 3e-4, 4e-4]
        schedule = [(lr, cand) for lr in lrs for cand in finalists]
        # Interleave candidates by LR rather than running one architecture nine times in a row.
        for lr, cand in schedule:
            model = model_map[cand]
            tag = f"{cand}-lr{lr:.0e}".replace("e-0", "e-")
            train = make_quality_train(tag, output_dir=QUALITY / "lr-sweep" / tag,
                                       max_tokens=sweep_tokens, adam_lr=lr, tokenizer=tokenizer)
            impl = "grouped" if "moe" in cand else "reference"
            tr = train_quality("lr-sweep", tag, model, train, data_path, moe_impl=impl,
                               baseline_used=baseline_used, baseline_temp=baseline_temp,
                               timeout_s=18000 if args.mode == "thorough" else 9000)
            trials.append(tr); finalist_rows.append(tr)
    summary["finalist_lr_sweep"] = [asdict(x) for x in finalist_rows]

    best_by_candidate: dict[str, Trial] = {}
    for tr in finalist_rows:
        cand = next((c for c in finalists if tr.name.startswith(c + "-lr")), None)
        if cand is None or tr.status != "ok":
            continue
        if cand not in best_by_candidate or eval_loss(tr) < eval_loss(best_by_candidate[cand]):
            best_by_candidate[cand] = tr
    chosen_cand = None
    chosen_lr = None
    if best_by_candidate:
        chosen_cand, chosen_trial = min(best_by_candidate.items(), key=lambda kv: eval_loss(kv[1]))
        # Recover LR from exact generated train config rather than parsing scientific notation.
        tr_cfg = load_yaml(CFG / f"train-{chosen_trial.name}.yaml")["train"]
        chosen_lr = float(tr_cfg["adam_lr"])
        summary["selected_finalist"] = {
            "architecture": chosen_cand, "adam_lr": chosen_lr,
            "eval_main_loss": eval_loss(chosen_trial), "tps": tps(chosen_trial),
            "note": "selected from proxy experiments only; not yet the final billion-scale architecture",
        }

    # ----- Longer top-two confirmation to catch short-screen ranking flips. -----
    confirmation: list[Trial] = []
    if best_by_candidate and prep.status == "ok":
        top2 = sorted(best_by_candidate.items(), key=lambda kv: eval_loss(kv[1]))[:2]
        confirm_tokens = 16_777_216 if args.mode == "thorough" else 4_194_304
        confirmation_seeds = [1337, 2027] if args.mode == "thorough" else [1337]
        candidate_lr: dict[str, float] = {}
        for cand, best_trial in top2:
            lr = float(load_yaml(CFG / f"train-{best_trial.name}.yaml")["train"]["adam_lr"])
            candidate_lr[cand] = lr
            for seed in confirmation_seeds:
                tag = f"{cand}-confirm-s{seed}"
                train = make_quality_train(
                    tag, output_dir=QUALITY / "confirmation" / tag,
                    max_tokens=confirm_tokens, adam_lr=lr, tokenizer=tokenizer, seed=seed,
                )
                impl = "grouped" if "moe" in cand else "reference"
                tr = train_quality(
                    "confirmation", tag, model_map[cand], train, data_path, moe_impl=impl,
                    baseline_used=baseline_used, baseline_temp=baseline_temp,
                    timeout_s=28800 if args.mode == "thorough" else 12000,
                )
                trials.append(tr); confirmation.append(tr)

        confirmation_stats: dict[str, Any] = {}
        for cand, _ in top2:
            xs = [x for x in confirmation if x.name.startswith(cand + "-confirm-s") and x.status == "ok" and math.isfinite(eval_loss(x))]
            if xs:
                confirmation_stats[cand] = {
                    "runs": len(xs),
                    "mean_eval_main_loss": statistics.mean(eval_loss(x) for x in xs),
                    "stdev_eval_main_loss": statistics.stdev(eval_loss(x) for x in xs) if len(xs) > 1 else 0.0,
                    "median_tps": statistics.median(tps(x) for x in xs),
                    "adam_lr": candidate_lr[cand],
                }
        summary["confirmation_stats"] = confirmation_stats
        if confirmation_stats:
            chosen_cand = min(confirmation_stats, key=lambda c: confirmation_stats[c]["mean_eval_main_loss"])
            chosen_lr = float(confirmation_stats[chosen_cand]["adam_lr"])
            summary["selected_after_confirmation"] = {
                "architecture": chosen_cand, "adam_lr": chosen_lr,
                **confirmation_stats[chosen_cand],
                "note": "proxy two-seed quality winner; speed remains a separate Pareto axis",
            }
    summary["confirmation"] = [asdict(x) for x in confirmation]

    # ----- Frontier-scale transfer sanity check for the confirmed proxy finalists. -----
    # A 220M screen is useful for breadth but cannot be assumed to transfer perfectly
    # to ~1B. Rebuild the confirmed candidates at frontier width with active FFN cost
    # re-matched, then run a short real-text check before the expensive final run.
    frontier_transfer: list[Trial] = []
    transfer_candidates = list((summary.get("confirmation_stats") or {}).keys())[:2]
    if transfer_candidates and prep.status == "ok":
        transfer_tokens = 4_194_304 if args.mode == "thorough" else 1_048_576
        for cand in transfer_candidates:
            model = make_frontier_variant(f"frontier-transfer-{cand}", cand, backend=chosen_backend)
            lr = float((summary.get("confirmation_stats") or {}).get(cand, {}).get("adam_lr") or chosen_lr or 3e-4)
            tag = f"frontier-{cand}"
            train = make_quality_train(
                tag, output_dir=QUALITY / "frontier-transfer" / tag,
                max_tokens=transfer_tokens, adam_lr=lr, tokenizer=tokenizer, seed=1337,
            )
            impl = "grouped" if "moe" in cand else "reference"
            tr = train_quality(
                "frontier-transfer", tag, model, train, data_path, moe_impl=impl,
                baseline_used=baseline_used, baseline_temp=baseline_temp,
                timeout_s=28800 if args.mode == "thorough" else 14400,
            )
            trials.append(tr); frontier_transfer.append(tr)
    summary["frontier_transfer"] = [asdict(x) for x in frontier_transfer]

    # ----- Optimizer quality/systems scout on the selected proxy architecture. -----
    # vNext1 established that the current single-GPU Muon implementation is much
    # slower than APOLLO-Mini. That is a systems result, not a quality verdict. Here
    # we measure whether Muon's sample-efficiency gain, if any, is remotely large
    # enough to justify its wall-clock cost before ruling it out for the long run.
    optimizer_rows: list[Trial] = []
    if chosen_cand and chosen_lr and prep.status == "ok":
        optimizer_tokens = 4_194_304 if args.mode == "thorough" else 1_048_576
        base_model = model_map[chosen_cand]
        optimizer_specs = [("apollo", "apollo_mini", None)]
        if args.mode == "thorough":
            optimizer_specs += [
                ("muon-5e-3", "muon_adamw", 5e-3),
                ("muon-1e-2", "muon_adamw", 1e-2),
                ("muon-2e-2", "muon_adamw", 2e-2),
            ]
        else:
            optimizer_specs += [("muon-1e-2", "muon_adamw", 1e-2)]
        for opt_tag, opt_kind, mu_lr in optimizer_specs:
            tag = f"{chosen_cand}-opt-{opt_tag}"
            train = make_quality_train(
                tag, output_dir=QUALITY / "optimizer" / tag,
                max_tokens=optimizer_tokens, adam_lr=chosen_lr, tokenizer=tokenizer,
                optimizer=opt_kind, seed=1337, muon_lr=(mu_lr or 0.01),
            )
            impl = "grouped" if "moe" in chosen_cand else "reference"
            tr = train_quality(
                "optimizer", tag, base_model, train, data_path, moe_impl=impl,
                baseline_used=baseline_used, baseline_temp=baseline_temp,
                timeout_s=18000 if args.mode == "thorough" else 9000,
            )
            trials.append(tr); optimizer_rows.append(tr)
    summary["optimizer_quality_scout"] = [asdict(x) for x in optimizer_rows]

    # ----- Real-data MTP ablation. Compare MAIN validation loss, not total loss. -----
    mtp_rows: list[Trial] = []
    if chosen_cand and chosen_lr and prep.status == "ok":
        base_model = model_map[chosen_cand]
        mtp_tokens = 8_388_608 if args.mode == "thorough" else 2_097_152
        mtp_seeds = [1337, 2027] if args.mode == "thorough" else [1337]
        for arch, weight in (("none", 0.0), ("low_rank", 0.12), ("deepseek", 0.12)):
            for seed in mtp_seeds:
                tag = f"{chosen_cand}-mtp-{arch}-s{seed}"
                m = with_mtp(tag, base_model, architecture=arch, weight=weight)
                trn = make_quality_train(
                    tag, output_dir=QUALITY / "mtp" / tag,
                    max_tokens=mtp_tokens, adam_lr=chosen_lr, tokenizer=tokenizer, seed=seed,
                )
                impl = "grouped" if "moe" in chosen_cand else "reference"
                tr = train_quality(
                    "mtp", tag, m, trn, data_path, moe_impl=impl,
                    baseline_used=baseline_used, baseline_temp=baseline_temp,
                    timeout_s=21600 if args.mode == "thorough" else 10800,
                )
                trials.append(tr); mtp_rows.append(tr)
    summary["mtp_ablation"] = [asdict(x) for x in mtp_rows]
    mtp_stats: dict[str, Any] = {}
    for arch in ("none", "low_rank", "deepseek"):
        xs = [x for x in mtp_rows if f"-mtp-{arch}-s" in x.name and x.status == "ok" and math.isfinite(eval_loss(x))]
        if xs:
            mtp_stats[arch] = {
                "runs": len(xs),
                "mean_eval_main_loss": statistics.mean(eval_loss(x) for x in xs),
                "stdev_eval_main_loss": statistics.stdev(eval_loss(x) for x in xs) if len(xs) > 1 else 0.0,
                "median_tps": statistics.median(tps(x) for x in xs),
                "mtp_loss_first_mean": statistics.mean((x.summary or {}).get("mtp_loss_first") or 0.0 for x in xs) if arch != "none" else None,
                "mtp_loss_last_mean": statistics.mean((x.summary or {}).get("mtp_loss_last") or 0.0 for x in xs) if arch != "none" else None,
            }
    summary["mtp_stats"] = mtp_stats

    # Save before report so a partially failed report/package does not lose the science.
    summary["trials"] = [asdict(x) for x in trials]
    summary["completed_at"] = now()
    (RUN / "campaign_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    report = build_report(summary)
    (RUN / "FRONTIER_VNEXT2_REPORT.md").write_text(report, encoding="utf-8")
    out_zip = package_results()
    print("\nAsterLM Frontier vNext2 campaign complete")
    print(f"Report:  {RUN / 'FRONTIER_VNEXT2_REPORT.md'}")
    print(f"Results: {out_zip}")


if __name__ == "__main__":
    main()
