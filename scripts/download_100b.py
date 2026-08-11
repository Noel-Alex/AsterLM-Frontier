#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import queue
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from asterlm.data.hf_stream import is_legacy_multistream_cursor, is_sequential_cursor  # noqa: E402

GIB = 2**30
DEFAULT_CORPUS_CONFIG = "configs/corpus/corpus_overtrain_100b.yaml"


@dataclass
class Tuning:
    readers: int = 10
    batch_rows: int = 16_384
    xet_concurrency: int = 24
    zstd_threads: int = 8
    zstd_buffer_mib: int = 8
    arrow_cpu_threads: int = 20
    arrow_io_threads: int = 16

    def copy(self) -> "Tuning":
        return Tuning(**self.__dict__)


@dataclass
class Stage:
    id: str
    command: list[str]
    source_id: str | None = None
    state_dir: Path | None = None
    target_tokens: int | None = None
    kind: str = "corpus"


@dataclass
class StageResult:
    stage_id: str
    returncode: int
    reason: str
    seconds: float
    attempts: int
    tuning: dict[str, Any]
    log: str


def human_tokens(value: float | int | None) -> str:
    if value is None:
        return "?"
    value = float(value)
    for suffix, scale in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(value) >= scale:
            return f"{value / scale:.2f}{suffix}"
    return f"{value:.0f}"


def human_rate(value: float | int | None) -> str:
    return f"{human_tokens(value)}/s" if value else "0 tok/s"


def human_seconds(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "?"
    seconds = int(seconds)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m"
    if minutes:
        return f"{minutes:d}m{seconds:02d}s"
    return f"{seconds:d}s"


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def read_net_bytes() -> tuple[int, int]:
    rx = tx = 0
    path = Path("/proc/net/dev")
    if not path.is_file():
        return 0, 0
    try:
        for line in path.read_text(encoding="utf-8").splitlines()[2:]:
            if ":" not in line:
                continue
            iface, values = line.split(":", 1)
            if iface.strip() == "lo":
                continue
            cols = values.split()
            rx += int(cols[0])
            tx += int(cols[8])
    except Exception:
        return 0, 0
    return rx, tx


def system_cpu_percent() -> float | None:
    try:
        import psutil
        return float(psutil.cpu_percent(interval=None))
    except Exception:
        return None


def process_tree_rss_gib(pid: int) -> float | None:
    try:
        import psutil
        root = psutil.Process(pid)
        total = 0
        for proc in [root] + root.children(recursive=True):
            try:
                total += proc.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return total / GIB
    except Exception:
        return None


def prepend_path(existing: str | None, *paths: Path) -> str:
    values = [str(path) for path in paths]
    if existing:
        values.append(existing)
    return os.pathsep.join(values)


def build_env(tuning: Tuning) -> dict[str, str]:
    env = dict(os.environ)
    env.pop("HF_XET_HIGH_PERFORMANCE", None)
    env["HF_XET_FIXED_DOWNLOAD_CONCURRENCY"] = str(tuning.xet_concurrency)
    env["HF_XET_CLIENT_MAX_IDLE_CONNECTIONS"] = "32"
    env.setdefault("HF_XET_NUM_CONCURRENT_RANGE_GETS", "16")
    env.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    env.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
    env["ASTERLM_HF_PARALLEL_STREAMS"] = str(tuning.readers)
    env["ASTERLM_PARQUET_BATCH_ROWS"] = str(tuning.batch_rows)
    env["ASTERLM_ARROW_CPU_THREADS"] = str(tuning.arrow_cpu_threads)
    env["ASTERLM_ARROW_IO_THREADS"] = str(tuning.arrow_io_threads)
    env["ASTERLM_ZSTD_LEVEL"] = "3"
    env["ASTERLM_ZSTD_THREADS"] = str(tuning.zstd_threads)
    env["ASTERLM_ZSTD_BUFFER_MIB"] = str(tuning.zstd_buffer_mib)
    env["ASTERLM_ARROW_TRIM_RSS_GIB"] = "16"
    env["ASTERLM_ARROW_TRIM_INTERVAL_SECONDS"] = "60"
    env["ASTERLM_TQDM"] = "0"
    env["ASTERLM_RUNTIME_HEARTBEAT_SECONDS"] = "2"
    env["ASTERLM_FORCE_IPV4"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONPATH"] = prepend_path(env.get("PYTHONPATH"), ROOT / "scripts" / "python_startup", ROOT / "src")
    return env


def load_corpus_config(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))["corpus"]
    return raw, list(raw["sources"])


def build_stages(args: argparse.Namespace) -> list[Stage]:
    config_path = ROOT / args.config
    raw, sources = load_corpus_config(config_path)
    output = ROOT / raw.get("output_dir", "data/corpus")
    stages: list[Stage] = []
    for source in sources:
        sid = str(source["id"])
        if args.only and sid not in args.only:
            continue
        if sid in args.skip:
            continue
        stages.append(Stage(
            id=f"corpus-{sid}", source_id=sid, target_tokens=int(source["target_tokens"]),
            state_dir=output / sid,
            command=[sys.executable, "scripts/materialize_corpus.py", "--config", args.config, "--only", sid,
                     "--max-retries", str(getattr(args, "materializer_retries", 2)), "--retry-base-seconds", "5",
                     "--retry-max-seconds", "90", "--checkpoint-seconds", str(getattr(args, "checkpoint_seconds", 300.0)),
                     "--checkpoint-documents", str(getattr(args, "checkpoint_documents", 100_000)), "--max-rss-gib", str(getattr(args, "max_rss_gib", 22.0))]
        ))
    return stages


def state_progress(stage: Stage) -> tuple[int | None, int | None, bool]:
    if stage.kind == "corpus" and stage.state_dir:
        state = read_json(stage.state_dir / "state.json")
        if state:
            return int(state.get("estimated_tokens", 0)), stage.target_tokens, bool(state.get("complete", False))
        return 0, stage.target_tokens, False
    if stage.kind == "stack" and stage.state_dir:
        total, found, complete = 0, False, True
        for state_path in stage.state_dir.glob("*/state.json"):
            state = read_json(state_path)
            if not state:
                continue
            found = True
            total += int(state.get("estimated_tokens", 0))
            complete = complete and bool(state.get("complete", False))
        return total, stage.target_tokens, complete if found else False
    return None, stage.target_tokens, False


def cursor_is_migratable(source_dir: Path) -> bool:
    state = read_json(source_dir / "state.json")
    if not state or not state.get("cursor_file"):
        return False
    cursor_path = source_dir / str(state["cursor_file"])
    try:
        with cursor_path.open("rb") as handle:
            cursor = pickle.load(handle)
        return is_sequential_cursor(cursor) or is_legacy_multistream_cursor(cursor)
    except Exception:
        return False


def migrate_one(source_dir: Path, *, show: bool = True) -> bool:
    if not cursor_is_migratable(source_dir):
        return False
    proc = subprocess.run([sys.executable, "scripts/migrate_legacy_cursor.py", str(source_dir)], cwd=ROOT,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=build_env(Tuning(readers=1)))
    text = proc.stdout.strip()
    if proc.returncode != 0:
        print(f"[cursor] {source_dir.name}: maintenance warning (materializer fallback remains available):")
        print(text)
        return False
    if show and text:
        print(text)
    return True


def migrate_stage(stage: Stage) -> None:
    if not stage.state_dir:
        return
    if stage.kind == "corpus":
        migrate_one(stage.state_dir)
    elif stage.state_dir.exists():
        for state_path in sorted(stage.state_dir.glob("*/state.json")):
            migrate_one(state_path.parent, show=False)


def stop_process_group(proc: subprocess.Popen[str], grace: float = 60.0) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGINT) if os.name == "posix" else proc.send_signal(signal.SIGINT)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace
    while proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.2)
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM) if os.name == "posix" else proc.terminate()
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL) if os.name == "posix" else proc.kill()
    except ProcessLookupError:
        pass


def runtime_path(stage: Stage) -> Path | None:
    return stage.state_dir / "runtime.json" if stage.kind == "corpus" and stage.state_dir else None


def format_dashboard(stage: Stage, runtime: dict[str, Any] | None, *, net_mbps: float,
                     cpu: float | None, process_rss: float | None, elapsed: float) -> str:
    if runtime:
        tokens = int(runtime.get("estimated_tokens", 0)); target = int(runtime.get("target_tokens", stage.target_tokens or 0))
        rate = float(runtime.get("token_rate_ema", 0.0) or 0.0); rss = runtime.get("rss_gib")
        arrow = runtime.get("arrow_allocated_gib"); phase = str(runtime.get("phase", "running"))
        percent = 100.0 * tokens / target if target else 0.0
        eta = (target - tokens) / rate if target and rate > 0 else None
        pieces = [f"{stage.source_id}: {human_tokens(tokens)}/{human_tokens(target)} {percent:5.1f}%", human_rate(rate),
                  f"RX {net_mbps:5.1f} Mbps", f"RSS {float(rss):.1f} GiB" if rss is not None else (f"RSS {process_rss:.1f} GiB" if process_rss is not None else "RSS ?")]
        if arrow is not None: pieces.append(f"Arrow {float(arrow):.1f} GiB")
        if cpu is not None: pieces.append(f"CPU {cpu:.0f}%")
        pieces += [f"ETA {human_seconds(eta)}", phase]
        return " | ".join(pieces)
    tokens, target, _ = state_progress(stage); percent = 100.0 * tokens / target if tokens is not None and target else 0.0
    pieces = [f"{stage.source_id}: {human_tokens(tokens)}/{human_tokens(target)} {percent:5.1f}%", f"RX {net_mbps:5.1f} Mbps"]
    if process_rss is not None: pieces.append(f"tree RSS {process_rss:.1f} GiB")
    if cpu is not None: pieces.append(f"CPU {cpu:.0f}%")
    pieces += [f"elapsed {human_seconds(elapsed)}", "starting / waiting for heartbeat"]
    return " | ".join(pieces)


def run_stage_once(stage: Stage, args: argparse.Namespace, tuning: Tuning, attempt: int) -> tuple[int, str, dict[str, Any] | None, str]:
    env = build_env(tuning); log_dir = ROOT / "data" / "download-logs-v11"; log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{stage.id}.log"; rp = runtime_path(stage)
    if rp: rp.unlink(missing_ok=True)
    print("\n" + "=" * 96); print(f"START {stage.id}  attempt={attempt}")
    print(f"readers={tuning.readers} batch={tuning.batch_rows:,} xet={tuning.xet_concurrency} zstd={tuning.zstd_threads} arrow={tuning.arrow_cpu_threads}/{tuning.arrow_io_threads}")
    print("$", " ".join(stage.command)); print(f"log: {log_path.relative_to(ROOT)}"); print("=" * 96)
    proc = subprocess.Popen(stage.command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, start_new_session=True)
    assert proc.stdout is not None
    lines: queue.Queue[str] = queue.Queue()
    def reader() -> None:
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n=== v11 attempt {attempt} at {time.ctime()} ===\n")
            for line in proc.stdout:
                log.write(line); log.flush(); stripped = line.rstrip()
                if stripped: lines.put(stripped)
    thread = threading.Thread(target=reader, daemon=True); thread.start()
    started = time.monotonic(); last_dashboard = 0.0; last_net_time = time.monotonic(); last_rx, _ = read_net_bytes(); net_mbps = 0.0
    last_runtime: dict[str, Any] | None = None
    try:
        while proc.poll() is None:
            while True:
                try: print(f"  │ {lines.get_nowait()}")
                except queue.Empty: break
            now = time.monotonic()
            if now - last_net_time >= 1.0:
                rx, _ = read_net_bytes(); dt = max(0.001, now - last_net_time); net_mbps = max(0.0, (rx-last_rx)*8.0/dt/1_000_000); last_rx = rx; last_net_time = now
            runtime = read_json(rp) if rp else None
            if runtime and int(runtime.get("pid", proc.pid)) == proc.pid: last_runtime = runtime
            if now - last_dashboard >= args.dashboard_seconds:
                print("  » " + format_dashboard(stage, last_runtime, net_mbps=net_mbps, cpu=system_cpu_percent(), process_rss=process_tree_rss_gib(proc.pid), elapsed=now-started)); last_dashboard = now
            if last_runtime:
                reference = float(last_runtime.get("last_record_unix") or last_runtime.get("started_at_unix") or time.time())
                no_record_for = time.time() - reference
                if no_record_for >= args.stall_seconds:
                    print(f"  ! {stage.id}: no source record for {no_record_for:.0f}s (RX {net_mbps:.1f} Mbps); checkpointing/retrying/defer.")
                    stop_process_group(proc); return 124, "stall", last_runtime, str(log_path)
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nShutdown requested; asking active stage to checkpoint..."); stop_process_group(proc); raise
    thread.join(timeout=2)
    while True:
        try: print(f"  │ {lines.get_nowait()}")
        except queue.Empty: break
    return int(proc.returncode or 0), "exit", last_runtime, str(log_path)


def adapt_after_memory(tuning: Tuning, runtime: dict[str, Any] | None) -> tuple[Tuning, str]:
    new = tuning.copy(); arrow = float((runtime or {}).get("arrow_allocated_gib", 0.0) or 0.0)
    if new.batch_rows > 4_096 and (arrow >= 2.0 or not runtime):
        new.batch_rows = max(4_096, new.batch_rows // 2); return new, f"batch rows {tuning.batch_rows:,} -> {new.batch_rows:,}"
    if new.readers > 6:
        new.readers = max(6, new.readers - 2); return new, f"readers {tuning.readers} -> {new.readers}"
    if new.xet_concurrency > 12:
        new.xet_concurrency = max(12, new.xet_concurrency - 4); return new, f"Xet concurrency {tuning.xet_concurrency} -> {new.xet_concurrency}"
    return new, "no safer automatic step remains"


def run_stage(stage: Stage, args: argparse.Namespace, base_tuning: Tuning) -> StageResult:
    tuning = base_tuning.copy(); network_attempts = 0; memory_adaptations = 0; started = time.monotonic(); last_log = ""
    while True:
        migrate_stage(stage); attempt = network_attempts + memory_adaptations + 1
        code, reason, runtime, last_log = run_stage_once(stage, args, tuning, attempt)
        if code == 0:
            return StageResult(stage.id, 0, "complete", time.monotonic()-started, attempt, tuning.__dict__.copy(), last_log)
        if code == 75 and memory_adaptations < args.memory_adaptations:
            memory_adaptations += 1; new_tuning, explanation = adapt_after_memory(tuning, runtime); print(f"[memory governor] {explanation}")
            if new_tuning.__dict__ == tuning.__dict__: break
            tuning = new_tuning; continue
        network_attempts += 1
        if network_attempts > args.source_retries:
            return StageResult(stage.id, code, reason, time.monotonic()-started, attempt, tuning.__dict__.copy(), last_log)
        if tuning.readers > 8: tuning.readers = 8
        elif tuning.xet_concurrency > 20: tuning.xet_concurrency = 20
        delay = min(60, 5 * (2 ** (network_attempts-1)))
        print(f"[retry] {stage.id}: code={code} reason={reason}; retry {network_attempts}/{args.source_retries} in {delay}s with readers={tuning.readers}, xet={tuning.xet_concurrency}")
        time.sleep(delay)
    return StageResult(stage.id, code, reason, time.monotonic()-started, attempt, tuning.__dict__.copy(), last_log)


def run_preflight(args: argparse.Namespace) -> None:
    print("Running one preflight for the complete campaign...")
    subprocess.run([sys.executable, "scripts/data_preflight.py", "--output", "data/data_preflight.json", "--minimum-free-gib", str(args.min_free_gib), "--require-auth"], cwd=ROOT, env=build_env(Tuning()), check=True)


def print_status(args: argparse.Namespace) -> int:
    stages = build_stages(args); print(f"{'source':20} {'tokens':>18} {'target':>12} {'done':>8} {'checkpoint':>12} status"); print("-"*90)
    for stage in stages:
        tokens, target, complete = state_progress(stage); checkpoint = "-"; status = "complete" if complete else "pending"
        if stage.kind == "corpus" and stage.state_dir:
            state = read_json(stage.state_dir / "state.json")
            if state: checkpoint = str(state.get("checkpoint_id", "-")); status = "complete" if complete else str(state.get("last_checkpoint_reason") or "pending")
        pct = (100.0*tokens/target) if tokens is not None and target else 0.0
        print(f"{str(stage.source_id):20} {human_tokens(tokens):>18} {human_tokens(target):>12} {pct:7.1f}% {checkpoint:>12} {status}")
    return 0


def migrate_all(args: argparse.Namespace) -> int:
    changed = 0
    for stage in build_stages(args):
        if not stage.state_dir: continue
        if stage.kind == "corpus": changed += int(migrate_one(stage.state_dir))
        elif stage.state_dir.exists():
            for state_path in sorted(stage.state_dir.glob("*/state.json")): changed += int(migrate_one(state_path.parent))
    print(f"Cursor maintenance complete; {changed} cursor(s) changed."); return 0


def run_campaign(args: argparse.Namespace) -> int:
    os.chdir(ROOT); run_preflight(args); stages = build_stages(args)
    tuning = Tuning(args.parallel_streams, args.parquet_batch_rows, args.xet_concurrency, args.zstd_threads, args.zstd_buffer_mib, args.arrow_cpu_threads, args.arrow_io_threads)
    print("\nAsterLM 100B downloader v11")
    print(f"fast profile: readers={tuning.readers}, Xet={tuning.xet_concurrency}, batch={tuning.batch_rows:,}; HP mode disabled")
    print(f"hard RSS ceiling={args.max_rss_gib:.1f} GiB; stall={args.stall_seconds:.0f}s; cursor maintenance=automatic")
    print("logs: data/download-logs-v11/\n")
    results: list[StageResult] = []; deferred: list[Stage] = []
    for stage in stages:
        tokens, target, complete = state_progress(stage)
        if complete and target and tokens is not None and tokens >= target:
            print(f"[skip complete] {stage.id}: {human_tokens(tokens)}/{human_tokens(target)}"); continue
        result = run_stage(stage, args, tuning); results.append(result)
        if result.returncode != 0:
            deferred.append(stage); print(f"[deferred] {stage.id}: code={result.returncode} reason={result.reason}; continuing.")
    if deferred and args.retry_deferred:
        print("\nRetrying deferred datasets once after the rest of the campaign...")
        still_failed: list[Stage] = []
        for stage in deferred:
            result = run_stage(stage, args, tuning); results.append(result)
            if result.returncode != 0: still_failed.append(stage)
        deferred = still_failed
    manifest = {"version":11, "finished_at_unix":time.time(), "config":args.config, "results":[item.__dict__ for item in results], "deferred_or_failed":[s.id for s in deferred]}
    out = ROOT / "data/download_run_v11.json"; tmp = out.with_suffix(".json.tmp"); tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"); os.replace(tmp, out)
    print(); print_status(args); print()
    if deferred:
        print("Campaign PARTIAL; run ./DOWNLOAD_100B.sh later. Remaining:")
        for stage in deferred: print(f"  - {stage.id}")
        return 2
    print("Selected campaign stages are complete."); return 0


def add_common_selection(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=DEFAULT_CORPUS_CONFIG)
    parser.add_argument("--skip", action="append", default=[])
    parser.add_argument("--only", action="append", default=[])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single-command, resumable AsterLM 100B corpus downloader"); sub = parser.add_subparsers(dest="command")
    run = sub.add_parser("run"); add_common_selection(run)
    run.add_argument("--parallel-streams", type=int, default=10); run.add_argument("--parquet-batch-rows", type=int, default=16_384); run.add_argument("--xet-concurrency", type=int, default=24)
    run.add_argument("--zstd-threads", type=int, default=8); run.add_argument("--zstd-buffer-mib", type=int, default=8); run.add_argument("--arrow-cpu-threads", type=int, default=20); run.add_argument("--arrow-io-threads", type=int, default=16)
    run.add_argument("--max-rss-gib", type=float, default=22.0); run.add_argument("--min-free-gib", type=float, default=150.0); run.add_argument("--stall-seconds", type=float, default=300.0)
    run.add_argument("--source-retries", type=int, default=2); run.add_argument("--materializer-retries", type=int, default=2); run.add_argument("--memory-adaptations", type=int, default=3)
    run.add_argument("--checkpoint-seconds", type=float, default=300.0); run.add_argument("--checkpoint-documents", type=int, default=100_000); run.add_argument("--dashboard-seconds", type=float, default=5.0)
    run.add_argument("--retry-deferred", action=argparse.BooleanOptionalAction, default=True)
    status = sub.add_parser("status"); add_common_selection(status)
    migrate = sub.add_parser("migrate"); add_common_selection(migrate)
    doctor = sub.add_parser("doctor"); doctor.add_argument("--min-free-gib", type=float, default=150.0)
    return parser


def main() -> int:
    parser = build_parser(); argv = sys.argv[1:]
    if not argv or argv[0].startswith("-"): argv = ["run", *argv]
    args = parser.parse_args(argv)
    if args.command == "status": return print_status(args)
    if args.command == "migrate": return migrate_all(args)
    if args.command == "doctor": run_preflight(args); return 0
    if args.command == "run": return run_campaign(args)
    parser.print_help(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
