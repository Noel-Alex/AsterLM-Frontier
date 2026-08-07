#!/usr/bin/env python3
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path.cwd()
MATERIALIZE = ROOT / "scripts/materialize_corpus.py"
RUNNER = ROOT / "scripts/run_100b_safe.py"
BACKUP_DIR = ROOT / ".asterlm-patch-backup-v8.1"

if not MATERIALIZE.is_file() or not RUNNER.is_file():
    raise SystemExit(
        "Run this from the AsterLM-Frontier repository root. "
        "Expected scripts/materialize_corpus.py and scripts/run_100b_safe.py."
    )

OLD_CREATE = '''def create_base_dataset(source: CorpusSource, revision: str) -> Any:
    # Importing zstandard before fsspec/datasets ensures .zst support is registered.
    import zstandard  # noqa: F401
    from datasets import load_dataset

    kwargs: dict[str, Any] = {
        "path": source.path,
        "name": source.name,
        "split": source.split,
        "streaming": True,
        "revision": revision,
    }
    if source.columns:
        kwargs["columns"] = source.columns
    return load_dataset(**kwargs)
'''

NEW_CREATE = '''def create_base_dataset(source: CorpusSource, revision: str) -> Any:
    # Importing zstandard before fsspec/datasets ensures .zst support is registered.
    import zstandard  # noqa: F401
    from datasets import load_dataset

    kwargs: dict[str, Any] = {
        "path": source.path,
        "name": source.name,
        "split": source.split,
        "streaming": True,
        "revision": revision,
    }
    if source.columns:
        kwargs["columns"] = source.columns

    # datasets==5.0.1 disables batch/fragment readahead in its Parquet generator.
    # On a high-latency remote filesystem that can leave a fast network mostly
    # idle. PyArrow's Parquet pre-buffer cache coalesces ranges and issues them
    # on its background I/O pool. Keep this process-local and configurable.
    parquet_prefetch = os.getenv("ASTERLM_PARQUET_PREFETCH", "1").strip().lower()
    if parquet_prefetch not in {"0", "false", "no", "off"}:
        try:
            import pyarrow as pa
            import pyarrow.dataset as pds

            io_threads = max(2, int(os.getenv("ASTERLM_ARROW_IO_THREADS", "24")))
            prefetch_limit = max(1, int(os.getenv("ASTERLM_PARQUET_PREFETCH_LIMIT", "8")))
            range_mib = max(16, int(os.getenv("ASTERLM_PARQUET_RANGE_MIB", "128")))

            pa.set_io_thread_count(io_threads)
            cache_options = pa.CacheOptions(
                prefetch_limit=prefetch_limit,
                range_size_limit=range_mib << 20,
            )
            kwargs["fragment_scan_options"] = pds.ParquetFragmentScanOptions(
                pre_buffer=True,
                cache_options=cache_options,
            )
        except (AttributeError, TypeError, ValueError):
            pass

    try:
        return load_dataset(**kwargs)
    except (TypeError, ValueError) as exc:
        if "fragment_scan_options" not in kwargs or "fragment_scan_options" not in str(exc):
            raise
        kwargs.pop("fragment_scan_options", None)
        return load_dataset(**kwargs)
'''

OLD_RUNNER_ENV = '''    env["ASTERLM_FORCE_IPV4"] = "0" if args.allow_ipv6 else "1"
    env["PYTHONUNBUFFERED"] = "1"
    command = [
'''

NEW_RUNNER_ENV = '''    env["ASTERLM_FORCE_IPV4"] = "0" if args.allow_ipv6 else "1"
    env["PYTHONUNBUFFERED"] = "1"

    # Keep one decoded Hugging Face input shard, but make the active remote
    # Parquet fragment aggressive. This creates a bounded raw-byte backlog
    # rather than restoring the old ten-decoded-stream RAM explosion.
    if args.network_mode in {"safe-fast", "fast"}:
        env.setdefault("ASTERLM_ARROW_IO_THREADS", "24")
        env.setdefault("ASTERLM_PARQUET_PREFETCH_LIMIT", "8")
        env.setdefault("ASTERLM_PARQUET_RANGE_MIB", "128")
    elif args.network_mode == "balanced":
        env.setdefault("ASTERLM_ARROW_IO_THREADS", "12")
        env.setdefault("ASTERLM_PARQUET_PREFETCH_LIMIT", "4")
        env.setdefault("ASTERLM_PARQUET_RANGE_MIB", "96")
    else:
        env.setdefault("ASTERLM_ARROW_IO_THREADS", "6")
        env.setdefault("ASTERLM_PARQUET_PREFETCH_LIMIT", "1")
        env.setdefault("ASTERLM_PARQUET_RANGE_MIB", "32")

    command = [
'''

OLD_RUNNER_PRINT = '''    print(f"Network mode:            {args.network_mode}")
    print("Dataset input shards:    1 active at a time (bounded Arrow memory)")
    print("Command-level relaunch:  disabled")
'''

NEW_RUNNER_PRINT = '''    print(f"Network mode:            {args.network_mode}")
    print("Dataset input shards:    1 active at a time (bounded Arrow memory)")
    print(f"Arrow I/O threads:        {env['ASTERLM_ARROW_IO_THREADS']}")
    print(f"Parquet prefetch ranges:  {env['ASTERLM_PARQUET_PREFETCH_LIMIT']}")
    print(f"Parquet range size:       {env['ASTERLM_PARQUET_RANGE_MIB']} MiB")
    print("Command-level relaunch:  disabled")
'''

def verify(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text and new not in text:
        raise SystemExit(
            f"REFUSING TO PATCH: expected {label} block was not found in {path}. "
            "No files were changed."
        )

def patch_file(path: Path, old: str, new: str, label: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if new in text:
        print(f"[already applied] {label}")
        return False
    if old not in text:
        raise SystemExit(f"REFUSING TO PATCH: expected {label} block disappeared from {path}.")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"[patched] {label}")
    return True

# Validate every expected block before modifying either file.
verify(MATERIALIZE, OLD_CREATE, NEW_CREATE, "create_base_dataset")
verify(RUNNER, OLD_RUNNER_ENV, NEW_RUNNER_ENV, "runner environment")
verify(RUNNER, OLD_RUNNER_PRINT, NEW_RUNNER_PRINT, "runner status output")

BACKUP_DIR.mkdir(parents=True, exist_ok=True)
for path in (MATERIALIZE, RUNNER):
    dst = BACKUP_DIR / path.name
    if not dst.exists():
        shutil.copy2(path, dst)
        print(f"[backup] {path} -> {dst}")

changed = False
changed |= patch_file(MATERIALIZE, OLD_CREATE, NEW_CREATE, "create_base_dataset")
changed |= patch_file(RUNNER, OLD_RUNNER_ENV, NEW_RUNNER_ENV, "runner Parquet tuning")
changed |= patch_file(RUNNER, OLD_RUNNER_PRINT, NEW_RUNNER_PRINT, "runner status output")

print("\\nRunning syntax compilation...")
subprocess.run(
    [sys.executable, "-m", "compileall", "-q", str(ROOT / "scripts"), str(ROOT / "src")],
    check=True,
)

print("\\nV8.1 applied successfully." if changed else "\\nV8.1 was already applied.")
print(f"Backups: {BACKUP_DIR}")
print("\\nRecommended run:")
print("  export HF_XET_FIXED_DOWNLOAD_CONCURRENCY=24")
print("  export HF_XET_CLIENT_MAX_IDLE_CONNECTIONS=32")
print("  unset HF_XET_HIGH_PERFORMANCE")
print("  ./RUN_100B_SAFE.sh --network-mode safe-fast")
