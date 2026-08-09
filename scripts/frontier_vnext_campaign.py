#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import yaml

ROOT = Path.cwd().resolve()
RUN_ROOT = ROOT / "runs" / "frontier-vnext"
CFG_ROOT = RUN_ROOT / "generated-configs"
LOG_ROOT = RUN_ROOT / "logs"
RESULT_ROOT = RUN_ROOT / "results"


@dataclass
class Trial:
    stage: str
    name: str
    status: str
    seconds: float
    command: list[str]
    env_delta: dict[str, str]
    result_path: str | None = None
    stdout_path: str | None = None
    error: str | None = None
    summary: dict[str, Any] | None = None


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def deep_copy(obj: Any) -> Any:
    return json.loads(json.dumps(obj))


def write_model(name: str, base_path: Path, changes: dict[str, Any]) -> Path:
    payload = deep_copy(load_yaml(base_path))
    model = payload.setdefault("model", payload if "model" not in payload else payload["model"])
    if "model" not in payload:
        payload = {"model": model}
    model.update(changes)
    path = CFG_ROOT / f"model-{name}.yaml"
    dump_yaml(path, payload)
    return path


def write_train(name: str, base_path: Path, changes: dict[str, Any]) -> Path:
    payload = deep_copy(load_yaml(base_path))
    train = payload.setdefault("train", payload if "train" not in payload else payload["train"])
    if "train" not in payload:
        payload = {"train": train}
    train.update(changes)
    path = CFG_ROOT / f"train-{name}.yaml"
    dump_yaml(path, payload)
    return path


def _json_or_none(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def summarize_profile(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not payload:
        return None
    out: dict[str, Any] = {"status": payload.get("status")}
    profile_summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    for key in ("median_seconds", "median_tokens_per_second"):
        if key in profile_summary:
            out[key] = profile_summary[key]
        elif key in payload:
            out[key] = payload[key]
    if "error" in payload:
        out["error"] = payload["error"]
    mem = profile_summary.get("final_memory") or payload.get("final_memory") or payload.get("memory")
    if isinstance(mem, dict):
        out["memory"] = {
            k: mem.get(k)
            for k in (
                "allocated_gib",
                "reserved_gib",
                "peak_allocated_gib",
                "peak_reserved_gib",
                "inactive_split_gib",
            )
            if k in mem
        }
    steps = payload.get("steps") or []
    measured = [s for s in steps if not s.get("warmup") and isinstance(s.get("tokens_per_second"), (int, float))]
    if measured:
        out["measured_tps_median"] = statistics.median(float(s["tokens_per_second"]) for s in measured)
        out["measured_tps_max"] = max(float(s["tokens_per_second"]) for s in measured)
        out["loss_last"] = measured[-1].get("loss")
        out["grad_norm_last"] = measured[-1].get("grad_norm")
    if "architecture" in payload:
        out["architecture"] = payload["architecture"]
    return out


def run_trial(
    stage: str,
    name: str,
    cmd: list[str],
    *,
    env_delta: dict[str, str] | None = None,
    timeout_s: int = 1800,
    result_path: Path | None = None,
) -> Trial:
    env_delta = dict(env_delta or {})
    env = os.environ.copy()
    env.update(env_delta)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"{stage}--{name}.log"
    started = time.perf_counter()
    status = "error"
    error = None
    returncode = None
    print(f"\n[{stage}] {name}", flush=True)
    print("$ " + " ".join(cmd), flush=True)
    try:
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(
                cmd,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        returncode = proc.returncode
        payload = _json_or_none(result_path) if result_path else None
        if payload and payload.get("status"):
            status = str(payload["status"])
        else:
            status = "ok" if returncode == 0 else "error"
        if returncode != 0 and status == "ok":
            status = "error"
        if payload and payload.get("error"):
            error = str(payload["error"])
        if returncode != 0 and not error:
            error = f"process returned {returncode}; see {log_path}"
    except subprocess.TimeoutExpired:
        status = "timeout"
        error = f"timeout after {timeout_s}s"
    except Exception as exc:
        status = "error"
        error = f"{type(exc).__name__}: {exc}"
    seconds = time.perf_counter() - started
    payload = _json_or_none(result_path) if result_path else None
    summary = summarize_profile(payload) if result_path else None
    print(f"=> {status} in {seconds:.1f}s" + (f" | {error}" if error else ""), flush=True)
    return Trial(
        stage=stage,
        name=name,
        status=status,
        seconds=seconds,
        command=cmd,
        env_delta=env_delta,
        result_path=str(result_path.relative_to(ROOT)) if result_path and result_path.exists() else None,
        stdout_path=str(log_path.relative_to(ROOT)),
        error=error,
        summary=summary,
    )


def profile(
    name: str,
    model: Path,
    train: Path,
    *,
    sequence: int,
    moe_impl: str,
    optimizer: str,
    precision: str = "transformer_engine_fp8",
    warmup: int = 1,
    steps: int = 3,
    timeout_s: int = 1800,
    extra_env: dict[str, str] | None = None,
    stage: str = "systems",
) -> Trial:
    path = RESULT_ROOT / f"{stage}--{name}.json"
    env = {
        "ASTER_MOE_IMPL": moe_impl,
        "PYTORCH_ALLOC_CONF": os.environ.get("PYTORCH_ALLOC_CONF", "expandable_segments:True"),
    }
    if extra_env:
        env.update(extra_env)
    cmd = [
        sys.executable,
        "scripts/profile_training.py",
        "--model",
        str(model),
        "--train-config",
        str(train),
        "--sequence",
        str(sequence),
        "--steps",
        str(steps),
        "--warmup",
        str(warmup),
        "--optimizer",
        optimizer,
        "--precision",
        precision,
        "--json",
        str(path),
    ]
    return run_trial(stage, name, cmd, env_delta=env, timeout_s=timeout_s, result_path=path)


def _status_ok(trial: Trial) -> bool:
    return trial.status == "ok" and bool(trial.summary)


def _trial_tps(trial: Trial) -> float:
    if not trial.summary:
        return 0.0
    return float(
        trial.summary.get("median_tokens_per_second")
        or trial.summary.get("measured_tps_median")
        or 0.0
    )


def _trial_peak(trial: Trial) -> float:
    try:
        return float((trial.summary or {}).get("memory", {}).get("peak_allocated_gib") or math.inf)
    except Exception:
        return math.inf


def discover_local_source(name: str) -> Path | None:
    data = ROOT / "data"
    if not data.is_dir():
        return None
    candidates: list[tuple[int, Path]] = []
    for p in data.rglob(name):
        if not p.is_dir():
            continue
        try:
            has_records = any(p.rglob("*.jsonl")) or any(p.rglob("*.jsonl.zst")) or any(p.rglob("*.jsonl.gz"))
        except Exception:
            has_records = False
        if not has_records:
            continue
        lower = str(p).lower()
        score = 0
        if "clean" in lower:
            score += 100
        if "studio" in lower:
            score += 20
        if "corpus" in lower:
            score += 10
        score -= len(p.parts)
        candidates.append((score, p))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], str(x[1])), reverse=True)
    return candidates[0][1]


def build_local_data_config() -> tuple[Path | None, dict[str, str]]:
    found = {name: discover_local_source(name) for name in (
        "fineweb_edu", "dclm", "finemath_4plus", "stack_edu", "cosmopedia_v2"
    )}
    paths = {k: str(v) for k, v in found.items() if v is not None}
    train_names = [n for n in ("fineweb_edu", "dclm", "finemath_4plus", "stack_edu") if found[n] is not None]
    if len(train_names) < 2 or found["cosmopedia_v2"] is None:
        return None, paths
    desired = {"fineweb_edu": 0.50, "dclm": 0.20, "finemath_4plus": 0.20, "stack_edu": 0.10}
    z = sum(desired[n] for n in train_names)
    sources = [
        {
            "path": str(found[n]),
            "split": "train",
            "text_field": "text",
            "weight": desired[n] / z,
        }
        for n in train_names
    ]
    payload = {
        "data": {
            "seed": 1337,
            "shuffle_buffer": 10000,
            "min_chars": 128,
            "max_chars": 200000,
            "quality_filters": True,
            "add_eos_between_documents": True,
            "sources": sources,
            "validation_sources": [
                {
                    "path": str(found["cosmopedia_v2"]),
                    "split": "train",
                    "text_field": "text",
                    "weight": 1.0,
                }
            ],
        }
    }
    path = CFG_ROOT / "data-local-proxy.yaml"
    dump_yaml(path, payload)
    return path, paths


def parse_metrics(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "metrics.jsonl"
    rows: list[dict[str, Any]] = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    eval_rows = [r for r in rows if "eval_main_loss" in r]
    train_rows = [r for r in rows if "main_loss" in r and "eval_main_loss" not in r]
    tps = [float(r["tokens_per_second"]) for r in train_rows if isinstance(r.get("tokens_per_second"), (int, float))]
    summary: dict[str, Any] = {
        "metrics_rows": len(rows),
        "eval_count": len(eval_rows),
        "final_eval_main_loss": eval_rows[-1].get("eval_main_loss") if eval_rows else None,
        "best_eval_main_loss": min((float(r["eval_main_loss"]) for r in eval_rows), default=None),
        "final_train_main_loss": train_rows[-1].get("main_loss") if train_rows else None,
        "median_tokens_per_second": statistics.median(tps) if tps else None,
        "max_tokens_seen": max((int(r.get("tokens_seen", 0)) for r in rows), default=0),
    }
    manifest = _json_or_none(run_dir / "run_manifest.json")
    if manifest:
        summary["architecture"] = manifest.get("architecture")
        summary["parameter_storage"] = manifest.get("parameter_storage")
    return summary


def latest_checkpoint(run_dir: Path) -> Path | None:
    # Used only for bookkeeping; screening checkpoints are not considered final artifacts.
    cands = sorted(
        [p for p in run_dir.rglob("*") if p.is_file() and p.suffix in {".pt", ".pth"}],
        key=lambda p: p.stat().st_mtime,
    )
    return cands[-1] if cands else None


def prune_proxy_checkpoints(run_dir: Path) -> None:
    for p in list(run_dir.rglob("*")):
        if p.is_file() and p.suffix in {".pt", ".pth", ".bin", ".safetensors"}:
            try:
                p.unlink()
            except Exception:
                pass
    for d in sorted([p for p in run_dir.rglob("*") if p.is_dir()], key=lambda p: len(p.parts), reverse=True):
        try:
            d.rmdir()
        except Exception:
            pass


def train_proxy(
    name: str,
    model_path: Path,
    train_path: Path,
    data_path: Path,
    *,
    moe_impl: str,
    timeout_s: int,
    stage: str,
) -> Trial:
    run_dir = Path((load_yaml(train_path).get("train") or {}).get("output_dir", RUN_ROOT / "proxy" / name))
    log_result = RESULT_ROOT / f"{stage}--{name}.json"
    env = {
        "ASTER_MOE_IMPL": moe_impl,
        "PYTORCH_ALLOC_CONF": os.environ.get("PYTORCH_ALLOC_CONF", "expandable_segments:True"),
    }
    cmd = [
        sys.executable,
        "scripts/train_pretrain.py",
        "--model",
        str(model_path),
        "--train",
        str(train_path),
        "--data",
        str(data_path),
    ]
    trial = run_trial(stage, name, cmd, env_delta=env, timeout_s=timeout_s)
    summary = parse_metrics(run_dir)
    summary["run_dir"] = str(run_dir)
    cp = latest_checkpoint(run_dir)
    summary["checkpoint_seen"] = str(cp) if cp else None
    trial.summary = summary
    if trial.status == "ok" and summary.get("max_tokens_seen", 0) <= 0:
        trial.status = "error"
        trial.error = "training returned successfully but no metrics/tokens were found"
    log_result.write_text(json.dumps({"trial": asdict(trial), "summary": summary}, indent=2, default=str), encoding="utf-8")
    trial.result_path = str(log_result.relative_to(ROOT))
    return trial


def make_proxy_model(name: str, variant: str, *, attnres: bool = False) -> Path:
    base = ROOT / "configs/model/aster_220m.yaml"
    payload = load_yaml(base)
    model = payload.get("model", payload)
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
        "use_block_attnres": attnres,
        "attnres_block_size": 4,
        "attention_train_backend": "sdpa",
    }
    if variant == "dense":
        changes.update({"ffn_type": "dense", "moe_first_dense_layers": 0})
    elif variant == "moe":
        changes.update({
            "ffn_type": "moe",
            "moe_every": 1,
            "moe_first_dense_layers": 2,
            "moe_num_experts": 8,
            "moe_top_k": 2,
            "moe_shared_experts": 1,
            "moe_expert_hidden": 704,
            "moe_aux_loss_weight": 1e-4,
            "moe_router_z_loss_weight": 1e-5,
            "moe_balance_strategy": "hybrid",
        })
    elif variant in {"latent_eff", "latent_acc"}:
        changes.update({
            "ffn_type": "latent_moe",
            "latent_moe_dim": int(model["d_model"]) // 4,
            "moe_every": 1,
            "moe_first_dense_layers": 2,
            "moe_num_experts": 32,
            "moe_top_k": 2 if variant == "latent_eff" else 8,
            "moe_shared_experts": 1,
            "moe_expert_hidden": 704,
            "moe_aux_loss_weight": 1e-4,
            "moe_router_z_loss_weight": 1e-5,
            "moe_balance_strategy": "hybrid",
        })
    elif variant == "gdn2_dense":
        pattern = ["gdn2" if kind == "kda" else kind for kind in _pattern_from_model(model)]
        changes.update({"ffn_type": "dense", "layer_pattern": pattern})
    else:
        raise ValueError(variant)
    return write_model(name, base, changes)


def _pattern_from_model(model: dict[str, Any]) -> list[str]:
    n_layers = int(model.get("n_layers", 24))
    ratio = int(model.get("kda_ratio", 3))
    explicit = model.get("layer_pattern")
    if explicit:
        return list(explicit)
    cycle = ["kda"] * ratio + ["latent"] if ratio > 0 else ["latent"]
    return [cycle[i % len(cycle)] for i in range(n_layers)]


def make_proxy_train(
    name: str,
    *,
    output_dir: Path,
    max_tokens: int,
    optimizer: str,
    fp8_recipe: str,
    eval_interval: int,
    warmup_steps: int,
    eval_batches: int,
) -> Path:
    base = ROOT / "configs/train/probe_memory_matrix.yaml"
    return write_train(
        name,
        base,
        {
            "output_dir": str(output_dir),
            "seed": 1337,
            "device": "cuda",
            "dtype": "bfloat16",
            "matmul_precision": "high",
            "compile": False,
            "precision_backend": "transformer_engine_fp8",
            "fp8_recipe": fp8_recipe,
            "fp8_format": "hybrid",
            "activation_offload": False,
            "sequence_length": 2048,
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 16,
            "max_steps": 1_000_000,
            "max_tokens": max_tokens,
            "optimizer": optimizer,
            "muon_lr": 0.01,
            "adam_lr": 3e-4,
            "warmup_steps": warmup_steps,
            "schedule_type": "constant",
            "weight_decay": 0.1,
            "max_grad_norm": 1.0,
            "log_interval": 4,
            "eval_interval": eval_interval,
            "eval_batches": eval_batches,
            "save_interval": 1_000_000,
            "keep_last_checkpoints": 1,
            "milestone_tokens": [],
            "diagnostic_interval": max(16, eval_interval),
            "save_diagnostic_bundle": False,
            "wandb_project": None,
            "tensorboard": False,
            "hub_repo_id": None,
            "hub_upload_final": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="One-shot AsterLM frontier research screening campaign")
    parser.add_argument("--mode", choices=["quick", "thorough"], default="thorough")
    parser.add_argument("--skip-architecture", action="store_true")
    parser.add_argument("--skip-long-context", action="store_true")
    args = parser.parse_args()

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    CFG_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)

    campaign_started = time.time()
    trials: list[Trial] = []
    notes: list[str] = []
    base_model = ROOT / "configs/model/aster_moe_frontier_893m_fp8.yaml"
    base_train = ROOT / "configs/train/probe_memory_matrix.yaml"
    if not base_model.is_file() or not base_train.is_file():
        raise SystemExit("Frontier base model/train configs are missing")

    # ------------------------------------------------------------------
    # Stage 1: corrected systems matrix. These runs intentionally change one
    # dimension at a time and all use the newly fixed block checkpoint path.
    # ------------------------------------------------------------------
    common_model = {
        "gradient_checkpointing": True,
        "checkpoint_segment_size": 1,
        "lm_loss_backend": "legacy_chunked",
        "attention_train_backend": "sdpa",
    }
    m_base = write_model("sys-base", base_model, common_model)
    t_delayed1024 = write_train("sys-delayed1024", base_train, {
        "fp8_recipe": "delayed", "fp8_amax_history_len": 1024, "activation_offload": False
    })

    trials.append(profile("moe-reference-1024", m_base, t_delayed1024, sequence=1024, moe_impl="reference", optimizer="apollo_mini"))
    trials.append(profile("moe-grouped-1024", m_base, t_delayed1024, sequence=1024, moe_impl="grouped", optimizer="apollo_mini"))

    m_no_ckpt = write_model("sys-no-checkpoint", base_model, {**common_model, "gradient_checkpointing": False})
    trials.append(profile("checkpoint-off-1024", m_no_ckpt, t_delayed1024, sequence=1024, moe_impl="grouped", optimizer="apollo_mini"))

    # Long-context memory is often dominated by checkpoint *boundaries*. Per-block
    # checkpointing retains one [B,T,D] input per layer. Screen wider checkpoint
    # segments now; this is a systems optimization, not an architecture change.
    segment_trials: list[tuple[int, Trial]] = []
    for segment_size in (1, 4, 8):
        m_segment = write_model(
            f"sys-checkpoint-seg{segment_size}",
            base_model,
            {**common_model, "checkpoint_segment_size": segment_size},
        )
        tr = profile(
            f"checkpoint-seg{segment_size}-2048",
            m_segment,
            t_delayed1024,
            sequence=2048,
            moe_impl="grouped",
            optimizer="apollo_mini",
            warmup=1,
            steps=2,
            stage="systems",
        )
        trials.append(tr)
        segment_trials.append((segment_size, tr))

    valid_segments = [(size, tr) for size, tr in segment_trials if _status_ok(tr)]
    chosen_segment = 4
    if valid_segments:
        # Context path values memory first; avoid a pathological speed regression.
        fastest = max(_trial_tps(tr) for _, tr in valid_segments)
        eligible = [
            (size, tr) for size, tr in valid_segments
            if _trial_tps(tr) >= 0.80 * fastest
        ] or valid_segments
        chosen_segment, segment_winner = min(
            eligible, key=lambda item: (_trial_peak(item[1]), -_trial_tps(item[1]))
        )
        notes.append(
            f"Checkpoint segment winner for long-context probes: {chosen_segment} "
            f"(peak={_trial_peak(segment_winner):.2f} GiB, tps={_trial_tps(segment_winner):.1f})"
        )
    else:
        notes.append("Checkpoint-segment screen failed; using segment size 4 as conservative default.")

    m_linear = write_model("sys-linear-ce", base_model, {**common_model, "lm_loss_backend": "torch_linear_ce"})
    trials.append(profile("linear-ce-1024", m_linear, t_delayed1024, sequence=1024, moe_impl="grouped", optimizer="apollo_mini"))

    t_delayed16 = write_train("sys-delayed16", base_train, {
        "fp8_recipe": "delayed", "fp8_amax_history_len": 16, "activation_offload": False
    })
    t_current = write_train("sys-current", base_train, {
        "fp8_recipe": "current", "activation_offload": False
    })
    trials.append(profile("fp8-delayed16-1024", m_linear, t_delayed16, sequence=1024, moe_impl="grouped", optimizer="apollo_mini"))
    trials.append(profile("fp8-current-1024", m_linear, t_current, sequence=1024, moe_impl="grouped", optimizer="apollo_mini"))

    # FLA exposes precision/performance controls for the triangular solves used by
    # delta-rule kernels. Keep these exploratory and do not silently select them
    # for quality runs until the resulting model-quality data justifies it.
    trials.append(profile(
        "fla-tril-tf32x3-1024", m_linear, t_delayed16, sequence=1024,
        moe_impl="grouped", optimizer="apollo_mini",
        extra_env={"FLA_TRIL_PRECISION": "tf32x3"},
    ))
    trials.append(profile(
        "fla-fastops-1024", m_linear, t_delayed16, sequence=1024,
        moe_impl="grouped", optimizer="apollo_mini",
        extra_env={"FLA_USE_FAST_OPS": "1"},
    ))

    for depth in (0, 1, 2):
        m_mtp = write_model(f"sys-mtp{depth}", base_model, {
            **common_model,
            "lm_loss_backend": "torch_linear_ce",
            "mtp_depth": depth,
            "mtp_loss_weight": 0.0 if depth == 0 else 0.12,
        })
        trials.append(profile(f"mtp-{depth}-1024", m_mtp, t_delayed16, sequence=1024, moe_impl="grouped", optimizer="apollo_mini"))

    trials.append(profile("optimizer-muon-1024", m_linear, t_delayed16, sequence=1024, moe_impl="grouped", optimizer="muon_adamw"))

    # Dense challenger with identical attention stack is an important sanity point:
    # sparse FFNs are not automatically faster on one GPU.
    m_dense = write_model("sys-frontier-dense", base_model, {
        **common_model,
        "lm_loss_backend": "torch_linear_ce",
        "ffn_type": "dense",
        "moe_first_dense_layers": 0,
    })
    trials.append(profile("dense-frontier-2048", m_dense, t_delayed16, sequence=2048, moe_impl="reference", optimizer="apollo_mini"))

    # Pick the fastest numerically-valid FP8 recipe for later performance/context probes.
    fp8_candidates = [t for t in trials if t.name in {
        "linear-ce-1024", "fp8-delayed16-1024", "fp8-current-1024"
    } and _status_ok(t)]
    chosen_recipe = "delayed"
    chosen_hist = 16
    if fp8_candidates:
        winner = max(fp8_candidates, key=_trial_tps)
        if winner.name == "fp8-current-1024":
            chosen_recipe, chosen_hist = "current", 16
        elif winner.name == "linear-ce-1024":
            chosen_recipe, chosen_hist = "delayed", 1024
        else:
            chosen_recipe, chosen_hist = "delayed", 16
        notes.append(f"FP8 screening throughput winner: {winner.name} ({_trial_tps(winner):.1f} tok/s)")
    else:
        notes.append("No alternate FP8 recipe completed; defaulting later probes to delayed/16.")

    preferred_train = write_train("preferred", base_train, {
        "fp8_recipe": chosen_recipe,
        "fp8_amax_history_len": chosen_hist,
        "activation_offload": False,
    })
    preferred_model = write_model("preferred", base_model, {
        "gradient_checkpointing": True,
        "checkpoint_segment_size": chosen_segment,
        "lm_loss_backend": "torch_linear_ce",
        "attention_train_backend": "sdpa",
    })

    # Operator profile on the corrected preferred path.
    kernel_result = RESULT_ROOT / "kernel-profile-1024.json"
    trials.append(run_trial(
        "kernel",
        "preferred-1024",
        [sys.executable, "scripts/frontier_vnext_kernel_profile.py", "--model", str(preferred_model), "--train", str(preferred_train), "--sequence", "1024", "--optimizer", "apollo_mini", "--precision", "transformer_engine_fp8", "--output", str(kernel_result)],
        env_delta={"ASTER_MOE_IMPL": "grouped", "PYTORCH_ALLOC_CONF": "expandable_segments:True"},
        timeout_s=1800,
        result_path=kernel_result,
    ))

    # ------------------------------------------------------------------
    # Stage 2: context feasibility. We intentionally measure both the sparse
    # frontier MoE and a dense challenger: on a 12GB card total expert storage can
    # cost more context than sparse activation saves. MTP2 is kept in the first
    # SDPA ladder; the extreme sparse-context ladders disable MTP to isolate the
    # mixer/activation-memory ceiling. Stage 4 later measures whether MTP earns its
    # training cost in quality.
    # ------------------------------------------------------------------
    if not args.skip_long_context:
        # Production-like current frontier: MTP2 + corrected checkpointing.
        for seq in (4096, 8192, 16384, 32768):
            tr = profile(
                f"moe-sdpa-mtp2-{seq}", preferred_model, preferred_train,
                sequence=seq, moe_impl="grouped", optimizer="apollo_mini",
                warmup=1 if seq <= 8192 else 0, steps=1,
                timeout_s=1200 if seq <= 8192 else 1800, stage="context",
            )
            trials.append(tr)
            if tr.status in {"oom", "timeout"}:
                break

        # Full-scale dense challenger with identical attention stack. Dense can be
        # the better *systems* choice on a single bandwidth/memory-constrained GPU.
        dense_context_model = write_model("context-dense", base_model, {
            "ffn_type": "dense",
            "moe_first_dense_layers": 0,
            "gradient_checkpointing": True,
            "checkpoint_segment_size": chosen_segment,
            "lm_loss_backend": "torch_linear_ce",
            "attention_train_backend": "sdpa",
        })
        for seq in (4096, 8192, 16384, 32768):
            tr = profile(
                f"dense-sdpa-mtp2-{seq}", dense_context_model, preferred_train,
                sequence=seq, moe_impl="reference", optimizer="apollo_mini",
                warmup=1 if seq <= 8192 else 0, steps=1,
                timeout_s=1200 if seq <= 8192 else 1800, stage="context",
            )
            trials.append(tr)
            if tr.status in {"oom", "timeout"}:
                break

        # Sparse global-attention experiment: exact local window + attention sinks
        # + strided global anchors. KDA/GDN recurrent layers still carry full-history
        # state, so the sparse MLA layers are complementary rather than the only
        # global-memory mechanism. This is a research backend, not claimed DSA.
        def make_flex_context(name: str, *, dense: bool) -> Path:
            changes = {
                "gradient_checkpointing": True,
                "checkpoint_segment_size": max(chosen_segment, 4),
                "lm_loss_backend": "torch_linear_ce",
                "attention_train_backend": "flex_window",
                "attention_train_window": 4096,
                "attention_train_global_stride": 512,
                "attention_flex_block_size": 128,
                "max_seq_len": 262144,
                "rope_scaling_type": "yarn",
                "rope_scaling_factor": 32.0,
                "rope_original_max_position": 8192,
                "mtp_depth": 0,
                "mtp_loss_weight": 0.0,
            }
            if dense:
                changes.update({"ffn_type": "dense", "moe_first_dense_layers": 0})
            return write_model(name, base_model, changes)

        for family, flex_model, moe_impl in (
            ("moe-flex-mtp0", make_flex_context("context-flex-moe", dense=False), "grouped"),
            ("dense-flex-mtp0", make_flex_context("context-flex-dense", dense=True), "reference"),
        ):
            for seq in (8192, 16384, 32768, 65536, 131072, 262144):
                tr = profile(
                    f"{family}-{seq}", flex_model, preferred_train,
                    sequence=seq, moe_impl=moe_impl, optimizer="apollo_mini",
                    warmup=1 if seq <= 16384 else 0, steps=1,
                    timeout_s=1500 if seq <= 32768 else 3000, stage="context",
                )
                trials.append(tr)
                if tr.status in {"oom", "timeout"}:
                    break

    # ------------------------------------------------------------------
    # Stage 3: real-data architecture screening. It is deliberately a two-stage
    # funnel: cheap screen of all candidates, then longer runs of dense + the best
    # non-dense candidates. This produces much more useful evidence than a dozen
    # microscopic one-off tests.
    # ------------------------------------------------------------------
    data_cfg, discovered = build_local_data_config()
    (RESULT_ROOT / "local-data-discovery.json").write_text(json.dumps(discovered, indent=2), encoding="utf-8")
    architecture_rows: list[dict[str, Any]] = []
    if args.skip_architecture:
        notes.append("Architecture training was explicitly skipped.")
    elif data_cfg is None:
        notes.append("Architecture screening skipped: could not find >=2 local train families plus cosmopedia_v2 validation.")
    elif not (ROOT / "artifacts/tokenizer.json").is_file():
        notes.append("Architecture screening skipped: artifacts/tokenizer.json is missing.")
    else:
        # Token budgets are exact multiples of 2048*16=32768 tokens/update.
        if args.mode == "quick":
            opt_tokens = 1_048_576       # 32 updates
            screen_tokens = 2_097_152    # 64 updates
            finalist_tokens = 8_388_608  # 256 updates
            timeout_train = 2 * 3600
        else:
            opt_tokens = 4_194_304       # 128 updates
            screen_tokens = 8_388_608    # 256 updates
            finalist_tokens = 33_554_432 # 1024 updates
            timeout_train = 8 * 3600

        # Optimizer screening on the same dense model/data.
        dense_proxy = make_proxy_model("proxy-dense", "dense")
        opt_results: list[Trial] = []
        for optim in ("apollo_mini", "muon_adamw"):
            train_cfg = make_proxy_train(
                f"optimizer-{optim}",
                output_dir=RUN_ROOT / "proxy" / f"optimizer-{optim}",
                max_tokens=opt_tokens,
                optimizer=optim,
                fp8_recipe=chosen_recipe,
                eval_interval=max(8, opt_tokens // 32768 // 4),
                warmup_steps=max(2, opt_tokens // 32768 // 16),
                eval_batches=8,
            )
            tr = train_proxy(
                f"optimizer-{optim}", dense_proxy, train_cfg, data_cfg,
                moe_impl="reference", timeout_s=timeout_train, stage="optimizer-screen",
            )
            trials.append(tr)
            opt_results.append(tr)

        valid_opts = [t for t in opt_results if t.status == "ok" and isinstance((t.summary or {}).get("final_eval_main_loss"), (int, float))]
        chosen_optimizer = "muon_adamw"
        if valid_opts:
            # Primary selection is fixed-token validation loss; speed is a tiebreaker.
            chosen = min(valid_opts, key=lambda t: (float(t.summary["final_eval_main_loss"]), -float(t.summary.get("median_tokens_per_second") or 0.0)))
            chosen_optimizer = chosen.name.removeprefix("optimizer-")
            notes.append(f"Proxy optimizer winner at fixed tokens: {chosen_optimizer} (eval_main_loss={chosen.summary['final_eval_main_loss']})")
        else:
            notes.append("Optimizer proxy had no valid eval result; using muon_adamw for architecture screen.")

        variants = [
            ("dense", "dense", False, "reference"),
            ("dense-attnres", "dense", True, "reference"),
            ("standard-moe", "moe", False, "grouped"),
            ("latentmoe-eff", "latent_eff", False, "grouped"),
            ("latentmoe-acc", "latent_acc", False, "grouped"),
        ]
        # GDN2 is only useful if the installed FLA exposes it. Capability script
        # records the exact reason when it is unavailable; try construction/training
        # here but let failure remain a negative result rather than aborting campaign.
        variants.append(("gdn2-dense", "gdn2_dense", False, "reference"))

        screen_results: list[Trial] = []
        for label, variant, attnres, moe_impl in variants:
            model_cfg = make_proxy_model(f"screen-{label}", variant, attnres=attnres)
            train_cfg = make_proxy_train(
                f"screen-{label}",
                output_dir=RUN_ROOT / "proxy" / f"screen-{label}",
                max_tokens=screen_tokens,
                optimizer=chosen_optimizer,
                fp8_recipe=chosen_recipe,
                eval_interval=max(16, screen_tokens // 32768 // 4),
                warmup_steps=max(4, screen_tokens // 32768 // 16),
                eval_batches=12 if args.mode == "thorough" else 8,
            )
            tr = train_proxy(label, model_cfg, train_cfg, data_cfg, moe_impl=moe_impl, timeout_s=timeout_train, stage="architecture-screen")
            trials.append(tr)
            screen_results.append(tr)
            architecture_rows.append({"phase": "screen", "variant": label, **(tr.summary or {}), "status": tr.status})

        valid_arch = [t for t in screen_results if t.status == "ok" and isinstance((t.summary or {}).get("final_eval_main_loss"), (int, float))]
        # Always retain dense as the control and escalate up to two best alternatives.
        finalists: list[Trial] = []
        dense_trial = next((t for t in valid_arch if t.name == "dense"), None)
        if dense_trial:
            finalists.append(dense_trial)
        alternatives = sorted(
            [t for t in valid_arch if t.name != "dense"],
            key=lambda t: float(t.summary["final_eval_main_loss"]),
        )[:2]
        finalists.extend(alternatives)

        variant_lookup = {label: (variant, attnres, moe_impl) for label, variant, attnres, moe_impl in variants}
        finalist_run_results: list[Trial] = []
        for selected in finalists:
            label = selected.name
            variant, attnres, moe_impl = variant_lookup[label]
            model_cfg = make_proxy_model(f"finalist-{label}", variant, attnres=attnres)
            train_cfg = make_proxy_train(
                f"finalist-{label}",
                output_dir=RUN_ROOT / "proxy" / f"finalist-{label}",
                max_tokens=finalist_tokens,
                optimizer=chosen_optimizer,
                fp8_recipe=chosen_recipe,
                eval_interval=max(32, finalist_tokens // 32768 // 8),
                warmup_steps=max(8, finalist_tokens // 32768 // 32),
                eval_batches=16 if args.mode == "thorough" else 8,
            )
            tr = train_proxy(label, model_cfg, train_cfg, data_cfg, moe_impl=moe_impl, timeout_s=max(timeout_train, 12 * 3600), stage="architecture-finalist")
            trials.append(tr)
            finalist_run_results.append(tr)
            architecture_rows.append({"phase": "finalist", "variant": label, **(tr.summary or {}), "status": tr.status})

        # MTP is a quality/sample-efficiency hypothesis, not a free training-speed
        # optimization. Test depth 0/1/2 on the best longer-run architecture using
        # identical language-model tokens, and retain both main loss and wall-clock.
        valid_finalists = [
            t for t in finalist_run_results
            if t.status == "ok" and isinstance((t.summary or {}).get("final_eval_main_loss"), (int, float))
        ]
        if valid_finalists:
            mtp_base_trial = min(valid_finalists, key=lambda t: float(t.summary["final_eval_main_loss"]))
            mtp_label = mtp_base_trial.name
            variant, attnres, moe_impl = variant_lookup[mtp_label]
            mtp_base_model = make_proxy_model(f"mtp-base-{mtp_label}", variant, attnres=attnres)
            mtp_tokens = 2_097_152 if args.mode == "quick" else 8_388_608
            for depth in (0, 1, 2):
                mtp_model = write_model(
                    f"mtp-quality-{mtp_label}-d{depth}",
                    mtp_base_model,
                    {
                        "mtp_depth": depth,
                        "mtp_rank": 256,
                        "mtp_loss_weight": 0.0 if depth == 0 else 0.12,
                    },
                )
                mtp_train = make_proxy_train(
                    f"mtp-quality-{mtp_label}-d{depth}",
                    output_dir=RUN_ROOT / "proxy" / f"mtp-quality-{mtp_label}-d{depth}",
                    max_tokens=mtp_tokens,
                    optimizer=chosen_optimizer,
                    fp8_recipe=chosen_recipe,
                    eval_interval=max(16, mtp_tokens // 32768 // 4),
                    warmup_steps=max(4, mtp_tokens // 32768 // 16),
                    eval_batches=12 if args.mode == "thorough" else 8,
                )
                tr = train_proxy(
                    f"{mtp_label}-mtp{depth}", mtp_model, mtp_train, data_cfg,
                    moe_impl=moe_impl, timeout_s=timeout_train, stage="mtp-quality",
                )
                trials.append(tr)
                architecture_rows.append({
                    "phase": "mtp-quality", "variant": mtp_label, "mtp_depth": depth,
                    **(tr.summary or {}), "status": tr.status,
                })

        (RESULT_ROOT / "architecture-comparison.json").write_text(
            json.dumps(architecture_rows, indent=2, default=str), encoding="utf-8"
        )
        # Screening checkpoints are intentionally not useful model releases and can
        # total several GB. Preserve metrics/manifests, not weights.
        for run_dir in (RUN_ROOT / "proxy").glob("*") if (RUN_ROOT / "proxy").is_dir() else []:
            prune_proxy_checkpoints(run_dir)

    # ------------------------------------------------------------------
    # Final machine-readable campaign summary.
    # ------------------------------------------------------------------
    summary = {
        "version": 1,
        "mode": args.mode,
        "started_unix": campaign_started,
        "finished_unix": time.time(),
        "wall_seconds": time.time() - campaign_started,
        "chosen_fp8_recipe": chosen_recipe,
        "chosen_fp8_amax_history_len": chosen_hist,
        "chosen_checkpoint_segment_size": chosen_segment,
        "notes": notes,
        "trials": [asdict(t) for t in trials],
    }
    (RUN_ROOT / "campaign_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    ok = sum(t.status == "ok" for t in trials)
    failed = len(trials) - ok
    print("\n" + "=" * 78)
    print("ASTER FRONTIER vNEXT CAMPAIGN COMPLETE")
    print(f"Trials: {len(trials)} | ok: {ok} | non-ok/negative-results: {failed}")
    print(f"Summary: {RUN_ROOT / 'campaign_summary.json'}")
    print("Non-ok trials are retained as data; the campaign intentionally continues.")
    print("=" * 78)


if __name__ == "__main__":
    main()
