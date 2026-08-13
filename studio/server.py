#!/usr/bin/env python
from __future__ import annotations

import argparse
import collections
import json
import math
import mimetypes
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import yaml

from asterlm.cloud import (
    build_gcp_launch_plan,
    build_modal_launch_plan,
    control_gcp_instance,
    control_modal_sandbox,
    dispatch_gcp_launch_plan,
    dispatch_modal_launch_plan,
    load_gcp_profile,
    load_modal_profile,
)
from asterlm.cuda_allocator import cuda_allocator_environment
from studio.providers import PROVIDER_CATALOG, provider_status
from studio.remote_contracts import build_contract, list_contracts, persist_contract
from studio.research_archive import ResearchArchive

ROOT = Path(__file__).resolve().parents[1]
STUDIO_ROOT = ROOT / "data" / "aster-studio"
LOG_ROOT = STUDIO_ROOT / "logs"
JOB_STATE = STUDIO_ROOT / "jobs.json"
SETTINGS_PATH = STUDIO_ROOT / "settings.json"
REMOTE_CONTRACT_ROOT = STUDIO_ROOT / "remote-contracts"
REMOTE_PLAN_ROOT = STUDIO_ROOT / "remote-plans"
REMOTE_JOB_ROOT = STUDIO_ROOT / "remote-jobs"
CATALOG_PATH = ROOT / "studio" / "catalog.yaml"
STATIC_ROOT = ROOT / "studio" / "static"
CAMPAIGN_PATH = ROOT / "configs" / "pretraining" / "frontier_100b_k3.yaml"
GIB = 2**30
_EXECUTION_BACKEND_CACHE: dict[str, Any] = {"updated": 0.0, "rows": []}
_EXECUTION_BACKEND_LOCK = threading.Lock()
_RESEARCH_ARCHIVE: ResearchArchive | None = None
_RESEARCH_ARCHIVE_LOCK = threading.Lock()

DEFAULT_SETTINGS: dict[str, Any] = {
    "download": {
        "parallel_streams": 10,
        "parquet_batch_rows": 16384,
        "xet_concurrency": 24,
        "arrow_cpu_threads": 20,
        "arrow_io_threads": 16,
        "zstd_threads": 8,
        "zstd_buffer_mib": 8,
        "max_rss_gib": 22.0,
        "stall_seconds": 90,
        "source_retries": 2,
        "materializer_retries": 2,
    },
    "training": {
        "checkpoint_tokens": 25000000,
        "keep_last_checkpoints": 6,
        "model": "configs/model/aster_k3_latentmoe_1p45b_a568m.yaml",
        "data": "data/clean-frontier/pretrain_data.yaml",
    },
    "ui": {
        "refresh_seconds": 2,
        "metric_points": 500,
    },
    "providers": {
        "preferred": "local",
        "require_cost_confirmation": True,
        "max_spend_usd_per_job": 30.0,
        "huggingface_namespaces": [],
        "gcp_profiles": ["google-credit"],
        "lightning_profiles": [],
        "skypilot_workspaces": [],
        "aws_profiles": [],
    },
}

SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
CONTRACT_ID = re.compile(r"^[0-9a-f]{20}$")


def provider_launch(contract_id: str, *, execute: bool) -> dict[str, Any]:
    if not CONTRACT_ID.fullmatch(contract_id):
        raise ValueError("Invalid remote contract id")
    contract_path = REMOTE_CONTRACT_ROOT / f"{contract_id}.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("contract_id") != contract_id:
        raise RuntimeError("Remote contract filename and identity differ")
    provider = contract.get("provider")
    profile_alias = str(contract["profile_alias"])
    if provider == "modal":
        profile = load_modal_profile(ROOT / "configs/providers/modal_boost.yaml", profile_alias)
        plan = build_modal_launch_plan(
            contract, profile, root=ROOT, contract_path=contract_path
        )
        result = dispatch_modal_launch_plan(plan, execute=execute)
    elif provider == "gcp":
        profile = load_gcp_profile(ROOT / "configs/providers/gcp_boost.yaml", profile_alias)
        plan = build_gcp_launch_plan(
            contract, profile, root=ROOT, contract_path=contract_path
        )
        result = dispatch_gcp_launch_plan(plan, execute=execute)
    else:
        raise ValueError(f"Provider {provider!r} does not have a dispatch adapter")
    atomic_json(REMOTE_PLAN_ROOT / f"{contract_id}-{provider}.json", plan)
    if execute and result.get("status") == "dispatched":
        remote_id = result.get("sandbox_id") or result.get("instance_name")
        if not isinstance(remote_id, str) or not SAFE_ID.fullmatch(remote_id):
            raise RuntimeError("Provider returned no safe remote job identity")
        atomic_json(
            REMOTE_JOB_ROOT / f"{remote_id}.json",
            {
                **result,
                "remote_id": remote_id,
                "contract_id": contract_id,
                "modal_environment": plan.get("modal_environment"),
                "created_at": time.time(),
            },
        )
    return result


def provider_control(remote_id: str, *, mode: str) -> dict[str, Any]:
    if not SAFE_ID.fullmatch(remote_id):
        raise ValueError("Invalid remote job id")
    job_path = REMOTE_JOB_ROOT / f"{remote_id}.json"
    job = json.loads(job_path.read_text(encoding="utf-8"))
    provider = str(job.get("provider") or "")
    if provider == "modal":
        result = control_modal_sandbox(
            profile_alias=str(job["profile_alias"]),
            modal_environment=str(job["modal_environment"]),
            sandbox_id=str(job["sandbox_id"]),
            mode=mode,
        )
    elif provider == "gcp":
        profile = load_gcp_profile(
            ROOT / "configs/providers/gcp_boost.yaml", str(job["profile_alias"])
        )
        selected = job.get("selected") or {}
        result = control_gcp_instance(
            profile=profile,
            instance_name=str(job["instance_name"]),
            zone=str(selected["zone"]),
            mode=mode,
        )
    else:
        raise ValueError(f"Provider {provider!r} does not have a control adapter")
    atomic_json(job_path, {**job, "last_control": result, "updated_at": time.time()})
    return result


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def deep_merge(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(base))
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def settings() -> dict[str, Any]:
    return deep_merge(DEFAULT_SETTINGS, load_json(SETTINGS_PATH, {}))


def validate_settings(value: dict[str, Any]) -> None:
    providers = value.get("providers") or {}
    preferred = str(providers.get("preferred", "local"))
    if preferred not in PROVIDER_CATALOG:
        raise ValueError(f"Unknown preferred provider: {preferred}")
    max_spend = float(providers.get("max_spend_usd_per_job", 0.0))
    if not math.isfinite(max_spend) or max_spend < 0:
        raise ValueError("Provider max_spend_usd_per_job must be a finite non-negative number")


def research_archive() -> ResearchArchive:
    global _RESEARCH_ARCHIVE
    expected_database = ROOT / "data" / "aster-studio" / "research.sqlite3"
    if _RESEARCH_ARCHIVE is None or _RESEARCH_ARCHIVE.root != ROOT.resolve():
        _RESEARCH_ARCHIVE = ResearchArchive(ROOT, expected_database)
    return _RESEARCH_ARCHIVE


def _background_research_reindex(archive: ResearchArchive) -> None:
    try:
        archive.reindex()
    finally:
        _RESEARCH_ARCHIVE_LOCK.release()


def refresh_research_archive(*, force: bool = False, max_age_seconds: float = 300.0) -> dict[str, Any]:
    archive = research_archive()
    summary = archive.summary()
    last_indexed = summary.get("last_indexed")
    if not force and last_indexed and time.time() - float(last_indexed) < max_age_seconds:
        summary["refresh_in_progress"] = _RESEARCH_ARCHIVE_LOCK.locked()
        return summary
    if force or not last_indexed:
        with _RESEARCH_ARCHIVE_LOCK:
            archive.reindex()
        summary = archive.summary()
        summary["refresh_in_progress"] = False
        return summary
    if _RESEARCH_ARCHIVE_LOCK.acquire(blocking=False):
        threading.Thread(
            target=_background_research_reindex,
            args=(archive,),
            name="aster-research-index",
            daemon=True,
        ).start()
    summary["refresh_in_progress"] = True
    return summary


def repo_path(value: str | Path, *, must_be_inside: bool = True) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if must_be_inside:
        try:
            path.relative_to(ROOT.resolve())
        except ValueError as exc:
            raise ValueError("Path must stay inside the AsterLM repository") from exc
    return path


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except Exception:
        return str(path)


def read_state(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def human_tokens(value: int | float | None) -> str:
    if value is None:
        return "—"
    x = float(value)
    for scale, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(x) >= scale:
            return f"{x / scale:.2f}{suffix}"
    return f"{x:.0f}"


def catalog() -> dict[str, Any]:
    return yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8"))


def corpus_config() -> dict[str, Any]:
    path = ROOT / "configs/corpus/corpus_overtrain_100b.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))["corpus"]


def stack_config() -> dict[str, Any]:
    path = ROOT / "configs/corpus/stack_edu_13b.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))["stack_edu"]


def dataset_status() -> list[dict[str, Any]]:
    cfg = corpus_config()
    output = ROOT / cfg.get("output_dir", "data/corpus-frontier-16b")
    rows: list[dict[str, Any]] = []
    for item in cfg["sources"]:
        sid = str(item["id"])
        state = read_state(output / sid / "state.json") or {}
        tokens = int(state.get("estimated_tokens", 0))
        target = int(item["target_tokens"])
        rows.append(
            {
                "id": sid,
                "label": sid,
                "tokens": tokens,
                "target": target,
                "percent": min(100.0, 100.0 * tokens / target) if target else 0.0,
                "complete": bool(state.get("complete", False)) and tokens >= target,
                "source_exhausted": bool(state.get("source_exhausted", False)),
                "checkpoint": state.get("checkpoint_id"),
                "reason": state.get("last_checkpoint_reason"),
                "path": rel(output / sid),
                "kind": "corpus",
            }
        )

    scfg = stack_config()
    sroot = ROOT / scfg.get("output_dir", "data/stack-edu-frontier-2p4b")
    total = 0
    target = sum(int(item["target_tokens"]) for item in scfg["languages"])
    language_rows = []
    for item in scfg["languages"]:
        language = str(item["name"])
        prefix = language.lower().replace("-", "_")
        state = read_state(sroot / prefix / "state.json") or {}
        value = int(state.get("estimated_tokens", 0))
        total += value
        language_rows.append(
            {
                "language": language,
                "tokens": value,
                "target": int(item["target_tokens"]),
                "complete": bool(state.get("complete", False)),
            }
        )
    rows.append(
        {
            "id": "stack_edu",
            "label": "Stack-Edu (retired)",
            "tokens": total,
            "target": target,
            "percent": min(100.0, 100.0 * total / target) if target else 0.0,
            "complete": total >= target,
            "source_exhausted": False,
            "checkpoint": None,
            "reason": "complete" if total >= target else "pending",
            "path": rel(sroot),
            "kind": "stack",
            "retired": True,
            "languages": language_rows,
        }
    )

    # Studio-created corpus sources use the same materializer state shape.
    custom_root = STUDIO_ROOT / "custom-corpus"
    if custom_root.exists():
        for state_path in sorted(custom_root.glob("*/state.json")):
            state = read_state(state_path) or {}
            source = state.get("source") or {}
            target = int(source.get("target_tokens", 0) or 0)
            tokens = int(state.get("estimated_tokens", 0))
            sid = str(source.get("id") or state_path.parent.name)
            rows.append(
                {
                    "id": sid,
                    "label": sid,
                    "tokens": tokens,
                    "target": target,
                    "percent": min(100.0, 100.0 * tokens / target) if target else 0.0,
                    "complete": bool(state.get("complete", False)) and (not target or tokens >= target),
                    "source_exhausted": bool(state.get("source_exhausted", False)),
                    "checkpoint": state.get("checkpoint_id"),
                    "reason": state.get("last_checkpoint_reason"),
                    "path": rel(state_path.parent),
                    "kind": "custom",
                }
            )
    return rows


def clean_corpora_status() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    plans = ROOT / "configs/studio/data-plans"
    if not plans.exists():
        return result
    for plan_path in sorted(plans.glob("*.yaml")):
        try:
            raw = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
            plan = raw.get("plan", raw)
            output = repo_path(str(plan.get("output", "")))
            state = read_state(output / "_studio_clean_state.json") or {}
            report = read_state(output / "studio_global_cleaning_report.json") or {}
            result.append(
                {
                    "name": str(plan.get("name") or plan_path.stem),
                    "plan": rel(plan_path),
                    "output": rel(output),
                    "generated_config": str(
                        plan.get(
                            "generated_config",
                            f"configs/studio/data/{plan_path.stem}_clean.yaml",
                        )
                    ),
                    "complete": bool(state.get("complete", False)),
                    "seen": int(state.get("seen", 0) or 0),
                    "kept": int(state.get("kept", 0) or 0),
                    "estimated_tokens": int(
                        report.get(
                            "estimated_tokens",
                            round(int(state.get("kept_chars", 0) or 0) / 4),
                        )
                    ),
                    "modified": (
                        (output / "_studio_clean_state.json").stat().st_mtime
                        if (output / "_studio_clean_state.json").exists()
                        else plan_path.stat().st_mtime
                    ),
                }
            )
        except Exception:
            continue
    return sorted(result, key=lambda item: item["modified"], reverse=True)


def disk_info() -> dict[str, Any]:
    usage = shutil.disk_usage(ROOT)
    return {
        "total_gib": usage.total / GIB,
        "used_gib": usage.used / GIB,
        "free_gib": usage.free / GIB,
        "percent": usage.used / usage.total * 100.0,
    }


def _optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() in {"N/A", "[N/A]", "NA", "NOT SUPPORTED"}:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def system_info() -> dict[str, Any]:
    result: dict[str, Any] = {"disk": disk_info()}
    try:
        import psutil
        mem = psutil.virtual_memory()
        result["memory"] = {
            "total_gib": mem.total / GIB,
            "used_gib": mem.used / GIB,
            "available_gib": mem.available / GIB,
            "percent": mem.percent,
        }
        result["cpu"] = {
            "percent": psutil.cpu_percent(interval=None),
            "load": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
            "count": psutil.cpu_count(),
        }
    except Exception as exc:
        result["memory"] = {"error": str(exc)}
        result["cpu"] = {"error": str(exc)}

    # Laptop GPUs frequently expose one or more telemetry fields as [N/A]
    # (especially power.limit). One unsupported optional metric must not turn
    # the entire GPU into "nvidia-smi unavailable".
    try:
        query = [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,utilization.gpu,"
            "temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem",
            "--format=csv,noheader,nounits",
        ]
        line = subprocess.check_output(
            query, text=True, stderr=subprocess.DEVNULL, timeout=3
        ).strip().splitlines()[0]
        fields = [item.strip() for item in line.split(",")]
        if not fields or not fields[0]:
            raise RuntimeError("nvidia-smi returned an empty GPU row")
        result["gpu"] = {
            "available": True,
            "name": fields[0],
            "memory_total_mib": _optional_float(fields[1] if len(fields) > 1 else None),
            "memory_used_mib": _optional_float(fields[2] if len(fields) > 2 else None),
            "utilization": _optional_float(fields[3] if len(fields) > 3 else None),
            "temperature_c": _optional_float(fields[4] if len(fields) > 4 else None),
            "power_w": _optional_float(fields[5] if len(fields) > 5 else None),
            "power_limit_w": _optional_float(fields[6] if len(fields) > 6 else None),
            "sm_clock_mhz": _optional_float(fields[7] if len(fields) > 7 else None),
            "mem_clock_mhz": _optional_float(fields[8] if len(fields) > 8 else None),
            "raw": line,
        }
    except Exception as exc:
        result["gpu"] = {"available": False, "error": str(exc)}
    return result


def execution_backend_status(ttl_seconds: float = 30.0) -> list[dict[str, Any]]:
    """Return a cached, evidence-separated view of training runtime readiness."""

    now = time.monotonic()
    with _EXECUTION_BACKEND_LOCK:
        cached = _EXECUTION_BACKEND_CACHE["rows"]
        if cached and now - float(_EXECUTION_BACKEND_CACHE["updated"]) < ttl_seconds:
            return list(cached)
        try:
            import torch

            from asterlm.training.execution import probe_execution_backends

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            capabilities = probe_execution_backends(
                device,
                lock_path=ROOT / "configs/research/upstream_sources_2026-08-10.yaml",
            )
            rows = [capability.to_dict() for capability in capabilities.values()]
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            rows = [
                {
                    "backend": "probe",
                    "importable": False,
                    "adapter_implemented": False,
                    "promoted": False,
                    "topology_supported": False,
                    "usable": False,
                    "blockers": [f"Capability probe failed: {type(exc).__name__}: {exc}"],
                }
            ]
        _EXECUTION_BACKEND_CACHE.update({"updated": now, "rows": rows})
        return list(rows)

def tail_lines(path: Path, limit: int = 300) -> list[str]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return list(collections.deque(handle, maxlen=max(1, min(limit, 5000))))


def metrics_for_run(run_path: Path, limit: int = 500) -> list[dict[str, Any]]:
    path = run_path / "metrics.jsonl"
    rows: list[dict[str, Any]] = []
    for line in tail_lines(path, limit):
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows


def checkpoint_list(run_path: Path) -> list[dict[str, Any]]:
    result = []
    for path in sorted(run_path.glob("checkpoint-*")):
        manifest = read_state(path / "checkpoint_manifest.json") or {}
        verification = read_state(
            run_path / "hub-verifications" / f"{path.name}.json"
        ) or {}
        local_bytes = sum(
            item.stat().st_size for item in path.rglob("*") if item.is_file()
        )
        result.append(
            {
                "name": path.name,
                "path": rel(path),
                "step": manifest.get("step"),
                "tokens_seen": manifest.get("tokens_seen"),
                "reason": manifest.get("reason"),
                "permanent": bool(manifest.get("permanent", False) or (path / "KEEP").exists()),
                "model_bytes": manifest.get("model_bytes"),
                "trainer_state_bytes": manifest.get("trainer_state_bytes"),
                "local_bytes": local_bytes,
                "hub_verified": (
                    verification.get("status") == "verified"
                    and verification.get("checkpoint") == path.name
                ),
                "hub_verified_files": verification.get("verified_file_count"),
                "hub_remote_commit": verification.get("remote_commit"),
            }
        )
    return result


def runs_status() -> list[dict[str, Any]]:
    runs_root = ROOT / "runs"
    rows: list[dict[str, Any]] = []
    if not runs_root.exists():
        return rows
    for path in sorted((p for p in runs_root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True):
        manifest = read_state(path / "run_manifest.json")
        experiment = read_state(path / "experiment.json")
        metrics = metrics_for_run(path, 120)
        checkpoints = checkpoint_list(path)
        if not manifest and not metrics and not checkpoints:
            continue
        latest = next(
            (
                row
                for row in reversed(metrics)
                if row.get("loss") is not None
                or row.get("tokens_per_second") is not None
            ),
            metrics[-1] if metrics else {},
        )
        rows.append(
            {
                "name": path.name,
                "path": rel(path),
                "modified": path.stat().st_mtime,
                "latest": latest,
                "architecture": (manifest or {}).get("architecture"),
                "checkpoint_count": len(checkpoints),
                "permanent_checkpoints": sum(item["permanent"] for item in checkpoints),
                "latest_checkpoint": checkpoints[-1] if checkpoints else None,
                "experiment": experiment,
                "run_id": (experiment or {}).get("run_id"),
                "stage": (experiment or {}).get("stage"),
                "status": (experiment or {}).get("status"),
                "completion_fraction": (experiment or {}).get("completion_fraction"),
                "wandb_url": ((experiment or {}).get("metrics") or {}).get("wandb_url"),
                "provider": ((experiment or {}).get("environment") or {}).get("provider"),
                "provider_profile_alias": ((experiment or {}).get("environment") or {}).get(
                    "provider_profile_alias"
                ),
                "parent_run_id": (experiment or {}).get("parent_run_id"),
            }
        )
    return rows[:100]


def _wandb_credentials_present() -> bool:
    if os.environ.get("WANDB_API_KEY"):
        return True
    for path in (Path.home() / ".netrc", Path.home() / "_netrc"):
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if re.search(r"(?im)^\s*machine\s+(api\.)?wandb\.ai\s*$", content):
            return True
    return False


def _git_worktree_clean() -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and not result.stdout.strip()


def _campaign_readiness(
    payload: dict[str, Any],
    *,
    providers: list[dict[str, Any]],
    jobs: list[dict[str, Any]],
) -> dict[str, Any]:
    data = payload.get("data") or {}
    rows: list[dict[str, Any]] = []

    def add(item_id: str, label: str, state: str, detail: str, *, blocking: bool = True) -> None:
        rows.append(
            {
                "id": item_id,
                "label": label,
                "state": state,
                "detail": detail,
                "blocking": blocking,
            }
        )

    expected = int(data.get("expected_materialized_tokens_after_downloads") or 0)
    materialized = int(data.get("current_materialized_tokens") or 0)
    acquisition_jobs = [
        job for job in jobs if job.get("action") == "download_corpus_sources"
    ]
    latest_acquisition = acquisition_jobs[0] if acquisition_jobs else None
    if materialized >= expected > 0:
        add("source_acquisition", "Raw source acquisition", "ready", f"{materialized:,} pinned raw tokens recorded")
    elif latest_acquisition and latest_acquisition.get("status") in {"running", "stopping"}:
        add("source_acquisition", "Raw source acquisition", "running", latest_acquisition.get("label", "materialization running"))
    elif latest_acquisition and latest_acquisition.get("status") == "failed":
        add(
            "source_acquisition",
            "Raw source acquisition",
            "blocked",
            "NVIDIA code repositories are gated; the current Hugging Face access request is awaiting review",
        )
    else:
        add("source_acquisition", "Raw source acquisition", "blocked", f"{materialized:,} / {expected:,} planned raw tokens materialized")

    clean_manifest = ROOT / str(data.get("clean_manifest") or "")
    clean_config = ROOT / str(data.get("clean_config") or "")
    clean_ready = clean_manifest.is_file() and clean_config.is_file()
    add(
        "clean_corpus",
        "Clean corpus seal",
        "ready" if clean_ready else "blocked",
        "decontaminated manifest and immutable data config present" if clean_ready else "final global cleaning, deduplication and decontamination have not completed",
    )

    tokenizer = ROOT / "artifacts" / "tokenizer.json"
    tokenizer_manifest = ROOT / "artifacts" / "tokenizer_manifest.json"
    tokenizer_ready = tokenizer.is_file() and tokenizer_manifest.is_file()
    add(
        "tokenizer",
        "Tokenizer seal",
        "ready" if tokenizer_ready else "blocked",
        "tokenizer and fertility/provenance manifest present" if tokenizer_ready else "final tokenizer must be trained and measured from the sealed corpus",
    )

    try:
        from asterlm.experiments import evaluate_promotion_gates

        decision = evaluate_promotion_gates(
            ROOT / "configs/experiments/promotion_gates.yaml", phase="stage1"
        )
        promotion_ready = decision.ready
        promotion_detail = (
            f"all {len(decision.required_gate_ids)} stage-1 gates passed"
            if decision.ready
            else f"{len(decision.blocking_gate_ids)} stage-1 evidence gates remain"
        )
    except Exception as exc:
        promotion_ready = False
        promotion_detail = f"gate ledger invalid: {type(exc).__name__}: {exc}"
    add(
        "promotion",
        "Promotion evidence",
        "ready" if promotion_ready else "blocked",
        promotion_detail,
    )

    hf = next((item for item in providers if item.get("id") == "huggingface_jobs"), {})
    add(
        "huggingface_auth",
        "Hugging Face authentication",
        "ready" if hf.get("authenticated") else "blocked",
        "provider-native token store detected" if hf.get("authenticated") else "run `hf auth login` before the durability round trip",
    )
    wandb_ready = _wandb_credentials_present()
    add(
        "wandb_auth",
        "Weights & Biases authentication",
        "ready" if wandb_ready else "blocked",
        "local W&B credential store detected" if wandb_ready else "run `wandb login` before the history-resume round trip",
    )

    git_clean = _git_worktree_clean()
    add(
        "git_pin",
        "Pinned source checkout",
        "ready" if git_clean else "blocked",
        "Git worktree is clean" if git_clean else "uncommitted implementation changes remain",
    )

    free_gib = shutil.disk_usage(ROOT).free / GIB
    budget_gib = float((payload.get("checkpointing") or {}).get("local_budget_gib") or 0)
    disk_ready = free_gib >= budget_gib + 20
    add(
        "local_storage",
        "Local recovery storage",
        "ready" if disk_ready else "blocked",
        f"{free_gib:.1f} GiB free; {budget_gib:.0f} GiB checkpoint cache plus 20 GiB safety margin required",
    )

    add(
        "hub_repo",
        "Private checkpoint repository",
        "input_required",
        "enter namespace/name in the launch control; Studio never exposes or stores a token",
    )
    blockers = [row["id"] for row in rows if row["blocking"] and row["state"] != "ready"]
    return {
        "ready": not blockers,
        "blocking_count": len(blockers),
        "blocking_ids": blockers,
        "items": rows,
    }


def training_campaign_status(
    *,
    providers: list[dict[str, Any]] | None = None,
    jobs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload = yaml.safe_load(CAMPAIGN_PATH.read_text(encoding="utf-8")) or {}
    runs = {row["path"]: row for row in runs_status()}
    completed = 0
    stages: list[dict[str, Any]] = []
    active: dict[str, Any] | None = None
    for raw in payload.get("stages", []):
        run = runs.get(str(raw["output_dir"]))
        latest = (run or {}).get("latest") or {}
        stage_tokens = min(int(latest.get("tokens_seen") or 0), int(raw["tokens"]))
        completed += stage_tokens
        row = {
            **raw,
            "completed_tokens": stage_tokens,
            "progress_fraction": stage_tokens / max(1, int(raw["tokens"])),
            "run": run,
            "freshness_seconds": (
                max(0.0, time.time() - float(latest.get("wall_time_unix")))
                if latest.get("wall_time_unix")
                else None
            ),
        }
        stages.append(row)
        if active is None and stage_tokens < int(raw["tokens"]):
            active = row
    target = int(payload.get("goal_tokens", 0))
    live = ((active or {}).get("run") or {}).get("latest") or {}
    throughput = live.get("tokens_per_second_ema") or live.get("tokens_per_second")
    provider_rows = providers if providers is not None else provider_status(settings().get("providers"))
    job_rows = jobs if jobs is not None else JOBS.list()
    readiness = _campaign_readiness(payload, providers=provider_rows, jobs=job_rows)
    return {
        **payload,
        "campaign_path": rel(CAMPAIGN_PATH),
        "completed_tokens": completed,
        "progress_fraction": completed / max(1, target),
        "active_stage": active,
        "stages": stages,
        "live": live,
        "eta_seconds": (
            (target - completed) / float(throughput)
            if throughput and completed < target
            else None
        ),
        "clean_manifest_ready": (ROOT / str(payload["data"]["clean_manifest"])).is_file(),
        "readiness": readiness,
    }


def diagnostic_matrices(limit: int = 100) -> list[dict[str, Any]]:
    """Return secret-safe summaries of nested systems/architecture matrices."""

    runs_root = ROOT / "runs"
    if not runs_root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in runs_root.rglob("matrix.json"):
        payload = read_state(path)
        if not isinstance(payload, dict):
            continue
        trials = payload.get("trials")
        trials = trials if isinstance(trials, list) else []
        aggregate = payload.get("aggregate")
        aggregate = aggregate if isinstance(aggregate, dict) else {}
        failures = [
            {
                "name": trial.get("name"),
                "status": trial.get("status"),
                "returncode": trial.get("returncode"),
            }
            for trial in trials
            if isinstance(trial, dict) and trial.get("status") not in {None, "ok"}
        ]
        rows.append(
            {
                "name": path.parent.name,
                "path": rel(path),
                "modified": path.stat().st_mtime,
                "created_utc": payload.get("created_utc"),
                "completed_utc": payload.get("completed_utc"),
                "status": "completed" if payload.get("completed_utc") else "in_progress",
                "git_commit": payload.get("git_commit"),
                "classification": payload.get("classification"),
                "protocol": payload.get("protocol") if isinstance(payload.get("protocol"), dict) else {},
                "aggregate": aggregate,
                "trial_count": len(trials),
                "successful_trials": sum(
                    isinstance(trial, dict) and trial.get("status") == "ok" for trial in trials
                ),
                "failures": failures,
            }
        )
    rows.sort(key=lambda row: float(row["modified"]), reverse=True)
    return rows[: max(1, min(int(limit), 500))]


class JobManager:
    def __init__(self) -> None:
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.jobs: dict[str, dict[str, Any]] = {}
        stored = load_json(JOB_STATE, {})
        if isinstance(stored, dict):
            self.jobs.update(stored)
        self._reconcile()

    def _reconcile(self) -> None:
        changed = False
        for job in self.jobs.values():
            if job.get("status") not in {"running", "stopping"}:
                continue
            pid = int(job.get("pid") or 0)
            alive = False
            if pid > 0:
                try:
                    os.kill(pid, 0)
                    alive = True
                except OSError:
                    alive = False
            if not alive:
                job["status"] = "unknown-exited"
                job["finished_at"] = job.get("finished_at") or time.time()
                changed = True
        if changed:
            self._save()

    def _save(self) -> None:
        atomic_json(JOB_STATE, self.jobs)

    def list(self) -> list[dict[str, Any]]:
        with self.lock:
            self._reconcile()
            return sorted(self.jobs.values(), key=lambda item: item.get("created_at", 0), reverse=True)

    def active_for_resource(self, resource: str) -> dict[str, Any] | None:
        for job in self.jobs.values():
            if job.get("resource") == resource and job.get("status") in {"running", "stopping"}:
                return job
        return None

    def start(
        self,
        *,
        label: str,
        action: str,
        command: list[str],
        env: dict[str, str] | None = None,
        resource: str = "cpu",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self.lock:
            blocking = self.active_for_resource(resource)
            if resource in {"gpu", "network"} and blocking is not None:
                raise RuntimeError(
                    f"Resource '{resource}' is already owned by {blocking['label']} ({blocking['id']}). "
                    "Stop that job first or wait for it to finish."
                )
            job_id = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
            log_path = LOG_ROOT / f"{job_id}.log"
            log_handle = log_path.open("ab", buffering=0)
            merged_env = dict(os.environ)
            if env:
                merged_env.update(env)
            if resource == "gpu":
                merged_env = cuda_allocator_environment(merged_env)
            proc = subprocess.Popen(
                command,
                cwd=ROOT,
                env=merged_env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            log_handle.close()
            job = {
                "id": job_id,
                "label": label,
                "action": action,
                "command": command,
                "pid": proc.pid,
                "status": "running",
                "returncode": None,
                "created_at": time.time(),
                "started_at": time.time(),
                "finished_at": None,
                "log": rel(log_path),
                "resource": resource,
                "metadata": metadata or {},
            }
            self.jobs[job_id] = job
            self._save()

            def waiter() -> None:
                code = proc.wait()
                with self.lock:
                    current = self.jobs.get(job_id)
                    if current:
                        current["returncode"] = code
                        current["status"] = "complete" if code == 0 else ("stopped" if code in {130, -2, -15} else "failed")
                        current["finished_at"] = time.time()
                        self._save()

            threading.Thread(target=waiter, name=f"aster-job-{job_id}", daemon=True).start()
            return job

    def stop(self, job_id: str, grace: float = 45.0) -> dict[str, Any]:
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                raise KeyError(job_id)
            if job.get("status") not in {"running", "stopping"}:
                return job
            job["status"] = "stopping"
            self._save()
            pid = int(job.get("pid") or 0)

        if pid <= 0:
            return job
        try:
            os.killpg(pid, signal.SIGINT)
        except ProcessLookupError:
            return job

        deadline = time.time() + grace
        while time.time() < deadline:
            try:
                os.kill(pid, 0)
            except OSError:
                return self.jobs[job_id]
            time.sleep(0.2)
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            return self.jobs[job_id]

        # Give a wedged child one final window. Studio's own trainer also treats
        # SIGTERM as a graceful-stop request; healthy runs should checkpoint and
        # exit before this expires.
        deadline = time.time() + 10.0
        while time.time() < deadline:
            try:
                os.kill(pid, 0)
            except OSError:
                return self.jobs[job_id]
            time.sleep(0.2)
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return self.jobs[job_id]


JOBS = JobManager()


def download_env() -> dict[str, str]:
    cfg = settings()["download"]
    env = dict(os.environ)
    env.pop("HF_XET_HIGH_PERFORMANCE", None)
    env["HF_XET_FIXED_DOWNLOAD_CONCURRENCY"] = str(cfg["xet_concurrency"])
    env["HF_XET_CLIENT_MAX_IDLE_CONNECTIONS"] = "32"
    env["HF_XET_NUM_CONCURRENT_RANGE_GETS"] = "16"
    env["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
    env["HF_HUB_ETAG_TIMEOUT"] = "30"
    env["ASTERLM_HF_PARALLEL_STREAMS"] = str(cfg["parallel_streams"])
    env["ASTERLM_PARQUET_BATCH_ROWS"] = str(cfg["parquet_batch_rows"])
    env["ASTERLM_ARROW_CPU_THREADS"] = str(cfg["arrow_cpu_threads"])
    env["ASTERLM_ARROW_IO_THREADS"] = str(cfg["arrow_io_threads"])
    env["ASTERLM_ZSTD_LEVEL"] = "3"
    env["ASTERLM_ZSTD_THREADS"] = str(cfg["zstd_threads"])
    env["ASTERLM_ZSTD_BUFFER_MIB"] = str(cfg["zstd_buffer_mib"])
    env["PYTHONUNBUFFERED"] = "1"
    return env


def sanitize_source(source_id: str) -> str:
    if not SAFE_ID.match(source_id):
        raise ValueError("Source ID may contain only letters, digits, dot, underscore and hyphen.")
    return source_id


def build_corpus_config(source_id: str, target_tokens: int, entry: dict[str, Any], *, main_root: bool = False) -> Path:
    source_id = sanitize_source(source_id)
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    source: dict[str, Any] = {
        "id": source_id,
        "path": entry["path"],
        "split": entry.get("split", "train"),
        "text_field": entry.get("text_field", "text"),
        "target_tokens": int(target_tokens),
        "shuffle_seed": int(entry.get("shuffle_seed", 1900)),
    }
    for key in (
        "name",
        "columns",
        "token_count_field",
        "min_chars",
        "max_chars",
        "require_fields",
        "revision",
    ):
        if entry.get(key) is not None:
            source[key] = entry[key]
    output_dir = "data/corpus-frontier-16b" if main_root else f"data/aster-studio/custom-corpus"
    raw = {
        "corpus": {
            "output_dir": output_dir,
            "shard_size_mb": 1024,
            "checkpoint_seconds": 300,
            "checkpoint_documents": 100000,
            "sources": [source],
        }
    }
    target = ROOT / "configs/studio/corpus" / f"{source_id}-{target_tokens}.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return target


def built_in_source_entry(source_id: str) -> dict[str, Any]:
    for item in corpus_config()["sources"]:
        if item["id"] == source_id:
            return item
    raise KeyError(source_id)


def create_clean_plan(payload: dict[str, Any]) -> Path:
    selected = payload.get("sources") or []
    if not selected:
        raise ValueError("Select at least one source")
    status_map = {item["id"]: item for item in dataset_status()}
    rows = []
    total = 0
    for item in selected:
        sid = sanitize_source(str(item["id"]))
        if sid == "stack_edu":
            raise ValueError("Stack-Edu is retired and cannot enter an active cleaning/training plan")
        current = status_map.get(sid)
        if current is None:
            raw_path = str(item.get("raw_path") or f"data/aster-studio/custom-corpus/{sid}")
            tokens = int(item.get("tokens") or 0)
        else:
            raw_path = current["path"]
            tokens = int(current["tokens"])
        if tokens <= 0 and not item.get("allow_empty", False):
            continue
        total += max(0, tokens)
        rows.append(
            {
                "id": sid,
                "raw_path": raw_path,
                "text_field": "text",
                "weight": float(item.get("weight") or max(tokens, 1)),
                "fim_rate": float(item.get("fim_rate", 0.5 if sid == "stack_edu" else 0.0)),
                "tokens": tokens,
                "optional": bool(item.get("optional", False)),
            }
        )
    if not rows:
        raise ValueError("No selected source has materialized tokens")

    name = sanitize_source(str(payload.get("name") or f"materialized-{round(total / 1e9)}b"))
    plan = {
        "plan": {
            "name": name,
            "output": str(payload.get("output") or f"data/clean-{name}"),
            "benchmarks": str(payload.get("benchmarks") or "data/decontamination-benchmarks"),
            "validation_fraction": float(payload.get("validation_fraction", 0.005)),
            "pii_mode": str(payload.get("pii_mode", "redact")),
            "near_distance": int(payload.get("near_distance", 3)),
            "audit_sample": int(payload.get("audit_sample", 10000)),
            "generated_config": str(payload.get("generated_config") or f"configs/studio/data/{name}_clean.yaml"),
            "sources": rows,
            "declared_materialized_tokens": total,
        }
    }
    path = ROOT / "configs/studio/data-plans" / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")
    return path


def start_action(action: str, payload: dict[str, Any]) -> dict[str, Any]:
    py = sys.executable
    env = download_env()

    if action == "modal_auth":
        alias = sanitize_source(str(payload["profile_alias"]))
        declared = {
            item
            for row in provider_status(settings().get("providers"))
            if row.get("id") == "modal"
            for item in row.get("declared_profiles", [])
        }
        if alias not in declared:
            raise ValueError("Modal profile alias must be declared in configs/providers/modal_boost.yaml")
        command_path = shutil.which("modal")
        if not command_path:
            raise RuntimeError("Modal CLI is not installed in the Studio environment")
        command = [
            command_path,
            "token",
            "new",
            "--profile",
            alias,
            "--no-activate",
            "--verify",
        ]
        return JOBS.start(
            label=f"Authorize Modal profile {alias}",
            action=action,
            command=command,
            resource=f"modal-auth-{alias}",
            metadata={"profile_alias": alias, "secrets_exposed": False},
        )

    if action == "download_source":
        source_id = sanitize_source(str(payload["source_id"]))
        target_tokens = int(payload["target_tokens"])
        if source_id == "stack_edu":
            raise ValueError("Stack-Edu is permanently retired from active acquisition")
        else:
            cat = catalog()["datasets"]
            entry = dict(payload.get("entry") or cat.get(source_id) or {})
            main_root = source_id in {"fineweb_edu", "dclm", "cosmopedia_v2", "finemath_4plus"}
            if not entry or not entry.get("path"):
                raise KeyError(
                    f"Unknown catalog source {source_id}; provide a custom dataset entry with at least path/split/text_field."
                )
            config = build_corpus_config(source_id, target_tokens, entry, main_root=main_root)
            command = [
                py,
                "scripts/materialize_corpus.py",
                "--config", rel(config),
                "--only", source_id,
                "--max-retries", str(settings()["download"]["materializer_retries"]),
                "--retry-base-seconds", "5",
                "--retry-max-seconds", "90",
                "--checkpoint-seconds", "300",
                "--checkpoint-documents", "100000",
                "--max-rss-gib", str(settings()["download"]["max_rss_gib"]),
            ]
        return JOBS.start(
            label=f"Download {source_id} → {human_tokens(target_tokens)}",
            action=action,
            command=command,
            env=env,
            resource="network",
            metadata={"source_id": source_id, "target_tokens": target_tokens},
        )

    if action == "download_corpus_sources":
        config = repo_path(str(payload["config"]))
        allowed_root = (ROOT / "configs" / "corpus").resolve()
        if allowed_root not in config.parents or config.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError("Corpus download config must be a YAML file under configs/corpus")
        source_ids = [sanitize_source(str(value)) for value in payload.get("source_ids") or []]
        if not source_ids:
            raise ValueError("Select at least one corpus source")
        raw = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
        declared = {
            str(source["id"])
            for source in (raw.get("corpus") or {}).get("sources", [])
            if source.get("id")
        }
        unknown = [value for value in source_ids if value not in declared]
        if unknown:
            raise ValueError(f"Unknown source ids for {rel(config)}: {', '.join(unknown)}")
        command = [
            py,
            "scripts/materialize_corpus.py",
            "--config",
            rel(config),
            "--max-retries",
            str(settings()["download"]["materializer_retries"]),
            "--retry-base-seconds",
            "5",
            "--retry-max-seconds",
            "90",
            "--checkpoint-seconds",
            "300",
            "--checkpoint-documents",
            "100000",
            "--max-rss-gib",
            str(settings()["download"]["max_rss_gib"]),
        ]
        for source_id in source_ids:
            command.extend(["--only", source_id])
        return JOBS.start(
            label=f"Materialize {len(source_ids)} pinned corpus sources",
            action=action,
            command=command,
            env=env,
            resource="network",
            metadata={"config": rel(config), "source_ids": source_ids},
        )

    if action == "download_profile":
        profile = str(payload["profile"])
        if profile not in {"benchmarks", "posttrain", "posttrain-modern", "posttrain-agent", "reasoning"}:
            raise ValueError("Only benchmark, post-training, and reasoning profiles are exposed here")
        command = [
            py,
            "scripts/download_data.py",
            "--profile", profile,
            "--validate-first",
            "--require-auth",
            "--network-mode", "safe-fast",
            "--max-retries", str(payload.get("max_retries", 10)),
            "--continue-on-error",
        ]
        return JOBS.start(
            label=f"Download {profile}",
            action=action,
            command=command,
            env=env,
            resource="network",
        )

    if action == "verify":
        target = repo_path(str(payload["path"]))
        command = [py, "scripts/verify_data_shards.py", rel(target)]
        if payload.get("only_last"):
            command.append("--only-last")
        return JOBS.start(label=f"Verify {rel(target)}", action=action, command=command, resource="disk")

    if action == "clean":
        plan = repo_path(str(payload["plan"]))
        command = [py, "scripts/studio_prepare_data.py", "--plan", rel(plan)]
        if payload.get("reset_existing"):
            command.append("--reset-existing")
        return JOBS.start(label=f"Clean {plan.stem}", action=action, command=command, resource="cpu")

    if action == "tokenizer":
        data = repo_path(str(payload["data"]))
        output = repo_path(str(payload.get("output", "artifacts/tokenizer.json")))
        command = [
            py, "scripts/train_tokenizer.py",
            "--data", rel(data),
            "--output", rel(output),
            "--vocab-size", str(int(payload.get("vocab_size", 32768))),
            "--documents", str(int(payload.get("documents", 1000000))),
        ]
        return JOBS.start(label="Train tokenizer", action=action, command=command, resource="cpu")

    if action == "capability_audit":
        command = [py, "scripts/aster_capability_audit.py", "--json", "data/aster-studio/capabilities.json"]
        if payload.get("smoke"):
            command.append("--smoke")
        return JOBS.start(label="Capability audit", action=action, command=command, resource="cpu")

    if action == "runtime_setup":
        profile = str(payload.get("profile", "core"))
        if profile not in {"kda", "apollo", "tracking", "torchao", "fp8", "reasoning", "baseline", "all"}:
            raise ValueError("Unknown runtime setup profile")
        command = [py, "scripts/studio_runtime_setup.py", profile]
        return JOBS.start(
            label=f"Install runtime: {profile}",
            action=action,
            command=command,
            resource="network",
            metadata={"profile": profile},
        )

    if action == "hardware_probe":
        command = [py, "scripts/hardware_probe.py", "--output", "runs/hardware-probe.json"]
        return JOBS.start(label="Hardware probe", action=action, command=command, resource="gpu")

    if action == "frontier_matrix":
        command = [
            py, "scripts/run_frontier_experiments.py",
            "--mode", str(payload.get("mode", "quick")),
            "--steps", str(int(payload.get("steps", 3))),
        ]
        return JOBS.start(label="Frontier VRAM matrix", action=action, command=command, resource="gpu")

    if action == "quality_ablations":
        command = [
            py, "scripts/run_quality_ablations.py",
            "--data", str(payload["data"]),
            "--tokens", str(int(payload.get("tokens", 100000000))),
            "--continue-on-error",
        ]
        return JOBS.start(label="Quality ablations", action=action, command=command, resource="gpu")

    if action == "preflight":
        command = [
            py, "scripts/training_preflight.py",
            "--model", str(payload["model"]),
            "--train", str(payload["train"]),
            "--data", str(payload["data"]),
            "--check-first-record",
            "--json", str(payload.get("json", "runs/studio-preflight.json")),
        ]
        if payload.get("checkpoint"):
            command += ["--checkpoint", str(payload["checkpoint"])]
        if payload.get("hub_repo"):
            command += ["--hub-repo", str(payload["hub_repo"])]
        return JOBS.start(label="Training preflight", action=action, command=command, resource="gpu")

    if action == "pretraining_campaign":
        hub_repo = str(payload.get("hub_repo") or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", hub_repo):
            raise ValueError("A public Hugging Face repository is required as namespace/name")
        command = [
            py,
            "scripts/run_pretraining_campaign.py",
            "--campaign",
            "configs/pretraining/frontier_100b_k3.yaml",
            "--hub-repo",
            hub_repo,
        ]
        entity = str(payload.get("wandb_entity") or "").strip()
        if entity:
            command.extend(["--wandb-entity", sanitize_source(entity)])
        if payload.get("verify_manifest_hashes", True):
            command.append("--verify-manifest-hashes")
        return JOBS.start(
            label="Aster K3 frozen 100B campaign",
            action=action,
            command=command,
            resource="gpu",
            metadata={"campaign": "aster-k3-100b", "hub_repo": hub_repo},
        )

    if action == "train_pretrain":
        command = [
            py, "scripts/studio_train.py",
            "--mode", "pretrain",
            "--model", str(payload["model"]),
            "--train", str(payload["train"]),
            "--data", str(payload["data"]),
        ]
        if payload.get("resume"):
            command += ["--resume", str(payload["resume"])]
        elif payload.get("init_checkpoint"):
            command += ["--init-checkpoint", str(payload["init_checkpoint"])]
        if payload.get("hub_repo"):
            command += ["--hub-repo", str(payload["hub_repo"])]
        return JOBS.start(label=f"Pretrain {Path(str(payload['train'])).stem}", action=action, command=command, resource="gpu")

    if action == "train_sft":
        command = [
            py, "scripts/studio_train.py",
            "--mode", "sft",
            "--model", str(payload["model"]),
            "--train", str(payload["train"]),
            "--data", str(payload["data"]),
        ]
        if payload.get("resume"):
            command += ["--resume", str(payload["resume"])]
        elif payload.get("checkpoint"):
            command += ["--checkpoint", str(payload["checkpoint"])]
        return JOBS.start(label="SFT", action=action, command=command, resource="gpu")

    if action == "dpo_reference":
        command = [
            py, "scripts/precompute_dpo_reference.py",
            "--checkpoint", str(payload["checkpoint"]),
            "--model", str(payload["model"]),
            "--tokenizer", str(payload.get("tokenizer", "artifacts/tokenizer.json")),
            "--input", str(payload["input"]),
            "--output", str(payload["output"]),
            "--max-length", str(int(payload.get("max_length", 2048))),
        ]
        return JOBS.start(label="Score DPO reference", action=action, command=command, resource="gpu")

    if action == "train_dpo":
        command = [
            py, "scripts/train_dpo.py",
            "--model", str(payload["model"]),
            "--train", str(payload["train"]),
            "--data", str(payload["data"]),
            "--max-length", str(int(payload.get("max_length", 2048))),
        ]
        if payload.get("resume"):
            command += ["--resume", str(payload["resume"])]
        elif payload.get("checkpoint"):
            command += ["--checkpoint", str(payload["checkpoint"])]
        return JOBS.start(label="DPO", action=action, command=command, resource="gpu")

    if action == "reasoning":
        command = [
            py, "scripts/run_reasoning_posttrain.py",
            "--model", str(payload["model"]),
            "--base-checkpoint", str(payload["checkpoint"]),
            "--reasoning", str(payload["reasoning"]),
        ]
        if payload.get("skip_prepare"):
            command.append("--skip-prepare")
        if payload.get("rl_stop_after") is not None:
            command += ["--rl-stop-after", str(int(payload["rl_stop_after"]))]
        return JOBS.start(label="Reasoning post-training", action=action, command=command, resource="gpu")

    if action == "eval_perplexity":
        command = [
            py, "scripts/evaluate_perplexity.py",
            "--checkpoint", str(payload["checkpoint"]),
            "--data", str(payload["data"]),
            "--sequence", str(int(payload.get("sequence", 8192))),
            "--batches", str(int(payload.get("batches", 32))),
        ]
        return JOBS.start(label="Perplexity evaluation", action=action, command=command, resource="gpu")
    if action == "eval_reasoning":
        command = [
            py, "scripts/evaluate_reasoning.py",
            "--model", str(payload["model"]),
            "--checkpoint", str(payload["checkpoint"]),
            "--data", str(payload["data"]),
            "--samples", str(int(payload.get("samples", 4))),
            "--limit", str(int(payload.get("limit", 100))),
        ]
        return JOBS.start(label="Reasoning evaluation", action=action, command=command, resource="gpu")


    if action == "benchmark":
        command = [
            py, "scripts/benchmark.py",
            "--checkpoint", str(payload["checkpoint"]),
            "--prompt-tokens", str(int(payload.get("prompt_tokens", 8192))),
            "--new-tokens", str(int(payload.get("new_tokens", 256))),
            "--cache-dtype", str(payload.get("cache_dtype", "hadamard_int4")),
        ]
        return JOBS.start(label="Inference benchmark", action=action, command=command, resource="gpu")

    if action == "benchmark_speculative":
        command = [
            py, "scripts/benchmark_speculative.py",
            "--checkpoint", str(payload["checkpoint"]),
            "--new-tokens", str(int(payload.get("new_tokens", 128))),
        ]
        return JOBS.start(label="Speculative benchmark", action=action, command=command, resource="gpu")
    if action == "benchmark_cache":
        command = [
            py, "scripts/benchmark_cache_quantization.py",
            "--tokens", str(int(payload.get("tokens", 32768))),
        ]
        return JOBS.start(label="KV-cache quantization benchmark", action=action, command=command, resource="gpu")

    if action == "export_osp":
        command = [
            py, "scripts/export_osp_merged.py",
            "--checkpoint", str(payload["checkpoint"]),
            "--output", str(payload["output"]),
        ]
        return JOBS.start(label="Export OSP-folded model", action=action, command=command, resource="gpu")

    if action == "export_torchao":
        command = [
            py, "scripts/export_torchao.py",
            "--checkpoint", str(payload["checkpoint"]),
            "--output", str(payload["output"]),
            "--mode", str(payload.get("mode", "int4")),
        ]
        return JOBS.start(label="Export TorchAO model", action=action, command=command, resource="gpu")

    if action == "hub_sync":
        command = [
            py, "scripts/sync_run_to_hub.py",
            "--run", str(payload["run"]),
            "--repo", str(payload["repo"]),
        ]
        return JOBS.start(label="Sync run to Hugging Face", action=action, command=command, resource="network")


    if action == "needle":
        command = [
            py, "scripts/needle_test.py",
            "--checkpoint", str(payload["checkpoint"]),
            "--lengths", str(payload.get("lengths", "8192,16384,32768")),
        ]
        return JOBS.start(label="Long-context needle test", action=action, command=command, resource="gpu")

    if action == "infer":
        command = [
            py, "scripts/infer.py",
            "--checkpoint", str(payload["checkpoint"]),
            "--prompt", str(payload["prompt"]),
            "--max-new-tokens", str(int(payload.get("max_new_tokens", 256))),
            "--cache-dtype", str(payload.get("cache_dtype", "hadamard_int4")),
        ]
        if payload.get("mtp_greedy"):
            command.append("--mtp-greedy")
        return JOBS.start(label="Inference playground", action=action, command=command, resource="gpu")

    raise ValueError(f"Unsupported action: {action}")


def config_files(kind: str) -> list[dict[str, Any]]:
    mapping = {
        "model": [ROOT / "configs/model", ROOT / "configs/studio/model"],
        "train": [ROOT / "configs/train", ROOT / "configs/studio/train"],
        "data": [ROOT / "configs/data", ROOT / "configs/studio/data"],
        "reasoning": [ROOT / "configs/reasoning", ROOT / "configs/studio/reasoning"],
        "corpus": [ROOT / "configs/corpus", ROOT / "configs/studio/corpus"],
    }
    roots = mapping.get(kind)
    if roots is None:
        raise ValueError("Unknown config kind")
    result: list[dict[str, Any]] = []
    for folder in roots:
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.yaml")):
            try:
                raw = yaml.safe_load(path.read_text(encoding="utf-8"))
                result.append({"path": rel(path), "name": path.stem, "config": raw})
            except Exception as exc:
                result.append({"path": rel(path), "name": path.stem, "error": str(exc)})
    return result


def save_studio_config(payload: dict[str, Any]) -> dict[str, Any]:
    kind = str(payload["kind"])
    name = sanitize_source(str(payload["name"]))
    raw = payload["config"]
    folder = ROOT / "configs/studio" / kind
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return {"path": rel(path), "config": raw}


def clone_config(payload: dict[str, Any]) -> dict[str, Any]:
    source = repo_path(str(payload["source"]))
    kind = str(payload["kind"])
    name = sanitize_source(str(payload["name"]))
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    return save_studio_config({"kind": kind, "name": name, "config": raw})


def generate_training_plan(payload: dict[str, Any]) -> dict[str, Any]:
    name = sanitize_source(str(payload["name"]))
    tokens = int(payload["tokens"])
    available = payload.get("available_tokens")
    command = [
        sys.executable,
        "scripts/studio_generate_training_plan.py",
        "--name", name,
        "--tokens", str(tokens),
        "--data", str(payload["data"]),
        "--checkpoint-tokens", str(int(payload.get("checkpoint_tokens", settings()["training"]["checkpoint_tokens"]))),
        "--keep-last", str(int(settings()["training"].get("keep_last_checkpoints", 6))),
    ]
    if available is not None:
        command += ["--available-tokens", str(int(available))]
    if payload.get("allow_repeat"):
        command.append("--allow-repeat")
    result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(result.stdout)
    plan_path = ROOT / "configs/studio/train" / f"{name}_plan.json"
    return load_json(plan_path, {"output": result.stdout})


def overview() -> dict[str, Any]:
    cap = load_json(STUDIO_ROOT / "capabilities.json", None)
    rows = dataset_status()
    clean = clean_corpora_status()
    jobs = JOBS.list()
    providers = provider_status(settings().get("providers"))
    return {
        "version": "1.1",
        "time": time.time(),
        "system": system_info(),
        "datasets": rows,
        "clean_corpora": clean,
        "raw_materialized_tokens": sum(int(item["tokens"]) for item in rows),
        "jobs": jobs,
        "runs": runs_status(),
        "capabilities": cap,
        "execution_backends": execution_backend_status(),
        "settings": settings(),
        "providers": providers,
        "training_campaign": training_campaign_status(providers=providers, jobs=jobs),
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "AsterLMStudio/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        # Keep the console readable; job logs are more useful than HTTP request spam.
        return

    def send_json(self, payload: Any, status: int = 200) -> None:
        data = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length > 4 * 1024 * 1024:
            raise ValueError("Request too large")
        raw = self.rfile.read(length) if length else b"{}"
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("JSON body must be an object")
        return value

    def serve_static(self, path: str) -> None:
        relative = "index.html" if path in {"", "/"} else path.lstrip("/")
        target = (STATIC_ROOT / relative).resolve()
        try:
            target.relative_to(STATIC_ROOT.resolve())
        except ValueError:
            self.send_error(404)
            return
        if not target.is_file():
            self.send_error(404)
            return
        mime = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        try:
            if path == "/api/overview":
                return self.send_json(overview())
            if path == "/api/catalog":
                return self.send_json(catalog())
            if path == "/api/jobs":
                return self.send_json(JOBS.list())
            if path == "/api/configs":
                kind = query.get("kind", ["model"])[0]
                return self.send_json(config_files(kind))
            if path == "/api/runs":
                return self.send_json(runs_status())
            if path == "/api/diagnostics":
                limit = int(query.get("limit", ["100"])[0])
                return self.send_json(diagnostic_matrices(limit))
            if path == "/api/research/summary":
                return self.send_json(refresh_research_archive())
            if path == "/api/research/trials":
                refresh_research_archive()
                return self.send_json(
                    research_archive().trials(
                        limit=int(query.get("limit", ["100"])[0]),
                        offset=int(query.get("offset", ["0"])[0]),
                        query=query.get("query", [""])[0],
                        backend=query.get("backend", [""])[0],
                        status=query.get("status", [""])[0],
                    )
                )
            if path == "/api/research/compare":
                refresh_research_archive()
                ids = [item for value in query.get("ids", []) for item in value.split(",") if item]
                return self.send_json(research_archive().compare(ids))
            if path == "/api/research/findings":
                refresh_research_archive()
                return self.send_json(
                    research_archive().findings(
                        limit=int(query.get("limit", ["100"])[0]),
                        offset=int(query.get("offset", ["0"])[0]),
                    )
                )
            if path == "/api/metrics":
                run = repo_path(query["run"][0])
                limit = int(query.get("limit", ["500"])[0])
                return self.send_json(metrics_for_run(run, limit))
            if path == "/api/checkpoints":
                run = repo_path(query["run"][0])
                return self.send_json(checkpoint_list(run))
            if path == "/api/job/log":
                job_id = query["id"][0]
                job = next((item for item in JOBS.list() if item["id"] == job_id), None)
                if not job:
                    raise KeyError(job_id)
                limit = int(query.get("limit", ["400"])[0])
                return self.send_json({"job": job, "lines": tail_lines(repo_path(job["log"]), limit)})
            if path == "/api/settings":
                return self.send_json(settings())
            if path == "/api/capabilities":
                report = load_json(STUDIO_ROOT / "capabilities.json", None)
                if report is None:
                    report = {"checks": catalog().get("capability_research", [])}
                return self.send_json(report)
            if path == "/api/execution-backends":
                return self.send_json(execution_backend_status())
            if path == "/api/providers":
                return self.send_json(provider_status(settings().get("providers")))
            if path == "/api/provider/contracts":
                return self.send_json(list_contracts(REMOTE_CONTRACT_ROOT))
            if path == "/api/provider/jobs":
                return self.send_json(
                    [
                        value
                        for item in sorted(REMOTE_JOB_ROOT.glob("*.json"), reverse=True)
                        if (value := load_json(item, None)) is not None
                    ]
                    if REMOTE_JOB_ROOT.is_dir()
                    else []
                )
            if path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
                return None
            return self.serve_static(path)
        except KeyError as exc:
            self.send_json({"error": f"Missing/not found: {exc}"}, 404)
        except Exception as exc:
            self.send_json({"error": str(exc), "type": type(exc).__name__}, 500)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            payload = self.read_json()
            if path == "/api/job/start":
                job = start_action(str(payload["action"]), payload.get("payload") or {})
                return self.send_json(job, 201)
            if path == "/api/job/stop":
                job = JOBS.stop(str(payload["id"]))
                return self.send_json(job)
            if path == "/api/settings":
                merged = deep_merge(settings(), payload)
                validate_settings(merged)
                atomic_json(SETTINGS_PATH, merged)
                return self.send_json(merged)
            if path == "/api/research/reindex":
                return self.send_json(refresh_research_archive(force=True))
            if path == "/api/provider/contract":
                current = settings()
                contract = build_contract(
                    payload,
                    root=ROOT,
                    policy=current.get("providers") or {},
                    providers=provider_status(current.get("providers")),
                )
                target = persist_contract(REMOTE_CONTRACT_ROOT, contract)
                contract["path"] = rel(target)
                return self.send_json(contract, 201)
            if path == "/api/provider/launch":
                return self.send_json(
                    provider_launch(
                        str(payload["contract_id"]), execute=bool(payload.get("execute", False))
                    ),
                    202 if payload.get("execute", False) else 200,
                )
            if path == "/api/provider/control":
                return self.send_json(
                    provider_control(
                        str(payload.get("remote_id") or payload.get("sandbox_id")),
                        mode=str(payload["mode"]),
                    )
                )
            if path == "/api/clean/plan":
                target = create_clean_plan(payload)
                return self.send_json(
                    {
                        "path": rel(target),
                        "plan": yaml.safe_load(target.read_text(encoding="utf-8")),
                    },
                    201,
                )
            if path == "/api/training/plan":
                return self.send_json(generate_training_plan(payload), 201)
            if path == "/api/config/save":
                return self.send_json(save_studio_config(payload), 201)
            if path == "/api/config/clone":
                return self.send_json(clone_config(payload), 201)
            if path == "/api/corpus/config":
                source_id = sanitize_source(str(payload["source_id"]))
                target_tokens = int(payload["target_tokens"])
                entry = dict(payload.get("entry") or catalog()["datasets"].get(source_id) or {})
                if not entry:
                    raise ValueError("Dataset entry is required for an unknown source")
                target = build_corpus_config(
                    source_id,
                    target_tokens,
                    entry,
                    main_root=bool(payload.get("main_root", False)),
                )
                return self.send_json({"path": rel(target), "config": yaml.safe_load(target.read_text(encoding="utf-8"))}, 201)
            self.send_json({"error": "Unknown API endpoint"}, 404)
        except KeyError as exc:
            self.send_json({"error": f"Missing/not found: {exc}"}, 400)
        except Exception as exc:
            self.send_json({"error": str(exc), "type": type(exc).__name__}, 500)


def self_test() -> None:
    assert CATALOG_PATH.is_file()
    assert (ROOT / "scripts/train_pretrain.py").is_file()
    assert (ROOT / "scripts/materialize_corpus.py").is_file()
    cat = catalog()
    assert "fineweb_edu" in cat["datasets"]
    cfg = corpus_config()
    assert any(item["id"] == "fineweb_edu" for item in cfg["sources"])
    _ = settings()
    _ = disk_info()
    print("AsterLM Studio self-test passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="AsterLM Studio local research control plane")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    STUDIO_ROOT.mkdir(parents=True, exist_ok=True)
    if not SETTINGS_PATH.exists():
        atomic_json(SETTINGS_PATH, DEFAULT_SETTINGS)

    if args.self_test:
        self_test()
        return

    address = (args.host, args.port)
    server = ThreadingHTTPServer(address, Handler)
    url = f"http://{args.host}:{args.port}/"
    print()
    print("AsterLM Studio")
    print("==============")
    print(f"Repository: {ROOT}")
    print(f"Local UI:   {url}")
    print("Ctrl+C stops the UI server; launched research jobs run in their own process groups.")
    print()

    if not args.no_open:
        threading.Timer(0.7, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print("\nStudio server stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
