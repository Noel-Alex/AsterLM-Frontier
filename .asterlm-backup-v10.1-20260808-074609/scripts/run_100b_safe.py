#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

GIB = 2**30


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def prepend_path(existing: str | None, *paths: Path) -> str:
    values = [str(path) for path in paths]
    if existing:
        values.append(existing)
    return os.pathsep.join(values)


def signal_process_group(process: subprocess.Popen[bytes], sig: signal.Signals) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, sig)
        else:
            process.send_signal(sig)
    except ProcessLookupError:
        pass


def wait_for_exit(process: subprocess.Popen[bytes], seconds: float) -> bool:
    deadline = time.monotonic() + max(0.0, seconds)
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.2)
    return process.poll() is not None


def bounded_stop(
    process: subprocess.Popen[bytes],
    *,
    interrupt_grace_seconds: float,
    terminate_grace_seconds: float,
    force: bool = False,
) -> int:
    """Stop the whole downloader process group with a finite escalation path."""

    if process.poll() is not None:
        return int(process.returncode or 0)

    if not force:
        print("\nSending SIGINT to the complete AsterLM downloader process group...", flush=True)
        signal_process_group(process, signal.SIGINT)
        if wait_for_exit(process, interrupt_grace_seconds):
            return int(process.returncode or 0)

    print("\nGrace period expired; sending SIGTERM to the complete process group...", flush=True)
    signal_process_group(process, signal.SIGTERM)
    if wait_for_exit(process, terminate_grace_seconds):
        return int(process.returncode or 0)

    print("\nProcess group still alive; sending SIGKILL.", flush=True)
    signal_process_group(process, signal.SIGKILL)
    try:
        return int(process.wait(timeout=10))
    except subprocess.TimeoutExpired:
        return 137


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the AsterLM 100B download with bounded disk, memory, network and shutdown behavior"
    )
    parser.add_argument("--min-free-gib", type=positive_float, default=float(os.getenv("MIN_FREE_GIB", "150")))
    parser.add_argument("--max-rss-gib", type=positive_float, default=float(os.getenv("MAX_RSS_GIB", "20")))
    parser.add_argument("--poll-seconds", type=positive_int, default=int(os.getenv("POLL_SECONDS", "15")))
    parser.add_argument(
        "--interrupt-grace-seconds",
        type=positive_float,
        default=float(os.getenv("INTERRUPT_GRACE_SECONDS", "45")),
    )
    parser.add_argument(
        "--terminate-grace-seconds",
        type=positive_float,
        default=float(os.getenv("TERMINATE_GRACE_SECONDS", "10")),
    )
    parser.add_argument("--profile", default=os.getenv("ASTERLM_PROFILE", "overtrain100"))
    parser.add_argument("--network-mode", default=os.getenv("ASTERLM_NETWORK_MODE", "balanced"))
    parser.add_argument(
        "--parallel-streams",
        type=positive_int,
        default=int(os.getenv("ASTERLM_HF_PARALLEL_STREAMS", "5")),
        help="Concurrent legacy Parquet child readers. Existing legacy cursors have at most 10 children.",
    )
    parser.add_argument(
        "--parquet-batch-rows",
        type=positive_int,
        default=int(os.getenv("ASTERLM_PARQUET_BATCH_ROWS", "16384")),
        help="Maximum decoded rows retained per active Parquet reader; smaller values reduce RSS.",
    )
    parser.add_argument(
        "--zstd-threads",
        type=positive_int,
        default=int(os.getenv("ASTERLM_ZSTD_THREADS", "8")),
        help="Zstd compression workers used to drain the materializer backlog.",
    )
    parser.add_argument(
        "--zstd-buffer-mib",
        type=positive_int,
        default=int(os.getenv("ASTERLM_ZSTD_BUFFER_MIB", "8")),
        help="Buffered uncompressed output handed to zstd in large chunks.",
    )
    parser.add_argument(
        "--allow-ipv6",
        action="store_true",
        help="Do not force IPv4 DNS results. The default avoids broken IPv6 SYN-SENT hangs.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)

    env = dict(os.environ)
    env["PYTHONPATH"] = prepend_path(
        env.get("PYTHONPATH"),
        repo_root / "scripts" / "python_startup",
        repo_root / "src",
    )
    env["ASTERLM_FORCE_IPV4"] = "0" if args.allow_ipv6 else "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["ASTERLM_HF_PARALLEL_STREAMS"] = str(args.parallel_streams)
    env["ASTERLM_PARQUET_BATCH_ROWS"] = str(args.parquet_batch_rows)
    env.setdefault("ASTERLM_ARROW_CPU_THREADS", "16")
    env.setdefault("ASTERLM_ARROW_IO_THREADS", "16")
    env.setdefault("ASTERLM_ARROW_TRIM_RSS_GIB", "10")
    env.setdefault("ASTERLM_ARROW_TRIM_INTERVAL_SECONDS", "10")
    # Keep compression quality reasonable, but feed zstd large chunks and let
    # more CPU workers drain the decoded backlog.
    env.setdefault("ASTERLM_ZSTD_LEVEL", "3")
    env["ASTERLM_ZSTD_THREADS"] = str(args.zstd_threads)
    env["ASTERLM_ZSTD_BUFFER_MIB"] = str(args.zstd_buffer_mib)
    command = [
        sys.executable,
        "scripts/download_data.py",
        "--profile",
        args.profile,
        "--require-auth",
        "--network-mode",
        args.network_mode,
        "--max-retries",
        "0",
        "--command-retries",
        "0",
        "--max-rss-gib",
        str(args.max_rss_gib),
    ]

    print(f"Repository:              {repo_root}")
    print(f"Disk safety floor:       {args.min_free_gib:.0f} GiB")
    print(f"Materializer RSS ceiling:{args.max_rss_gib:.1f} GiB")
    print(f"Force IPv4:              {not args.allow_ipv6}")
    print(f"Network mode:            {args.network_mode}")
    print(f"Concurrent HF readers:   {args.parallel_streams}")
    print(f"Parquet batch rows:      {args.parquet_batch_rows:,}")
    print(f"Arrow CPU/I/O threads:   {env['ASTERLM_ARROW_CPU_THREADS']}/{env['ASTERLM_ARROW_IO_THREADS']}")
    print(f"Arrow trim threshold:    {env['ASTERLM_ARROW_TRIM_RSS_GIB']} GiB")
    print(f"Zstd level/threads:      {env['ASTERLM_ZSTD_LEVEL']}/{env['ASTERLM_ZSTD_THREADS']}")
    print(f"Zstd write buffer:       {args.zstd_buffer_mib} MiB")
    print("Prefetch policy:         one speculative row per active HF child")
    print("Command-level relaunch:  disabled")
    print("$", " ".join(command), flush=True)

    # A dedicated session gives us one process group containing download_data and
    # every materializer it launches. Terminal Ctrl-C reaches this supervisor;
    # the supervisor then forwards it to the complete child group.
    process = subprocess.Popen(command, cwd=repo_root, env=env, start_new_session=True)

    requested_signals: list[int] = []

    def request_stop(signum: int, _frame: object) -> None:
        requested_signals.append(signum)
        if len(requested_signals) == 1:
            print("\nShutdown requested; preserving a checkpoint when possible...", flush=True)
        else:
            print("\nSecond shutdown request received; escalating immediately...", flush=True)

    old_int = signal.signal(signal.SIGINT, request_stop)
    old_term = signal.signal(signal.SIGTERM, request_stop)
    next_disk_check = 0.0
    disk_stop = False

    try:
        while process.poll() is None:
            if requested_signals:
                code = bounded_stop(
                    process,
                    interrupt_grace_seconds=args.interrupt_grace_seconds,
                    terminate_grace_seconds=args.terminate_grace_seconds,
                    force=len(requested_signals) > 1,
                )
                return 130 if requested_signals[0] == signal.SIGINT else (143 if code == 0 else code)

            now = time.monotonic()
            if now >= next_disk_check:
                free_gib = shutil.disk_usage(repo_root).free / GIB
                print(
                    f"\rFree disk: {free_gib:6.1f} GiB | floor: {args.min_free_gib:.0f} GiB | "
                    f"materializer RSS ceiling: {args.max_rss_gib:.1f} GiB",
                    end="",
                    flush=True,
                )
                if free_gib < args.min_free_gib:
                    print("\nDisk safety floor reached; requesting a cursor + shard checkpoint.", flush=True)
                    disk_stop = True
                    bounded_stop(
                        process,
                        interrupt_grace_seconds=args.interrupt_grace_seconds,
                        terminate_grace_seconds=args.terminate_grace_seconds,
                    )
                    return 75
                next_disk_check = now + args.poll_seconds
            time.sleep(0.2)

        print()
        return int(process.returncode or 0)
    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)
        if process.poll() is None and not disk_stop:
            bounded_stop(
                process,
                interrupt_grace_seconds=args.interrupt_grace_seconds,
                terminate_grace_seconds=args.terminate_grace_seconds,
                force=True,
            )


if __name__ == "__main__":
    raise SystemExit(main())
