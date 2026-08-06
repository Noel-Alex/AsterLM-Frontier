#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any


def version(module_name: str) -> tuple[str | None, str | None]:
    try:
        module = __import__(module_name)
        return str(getattr(module, "__version__", "unknown")), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the long-running AsterLM data-download environment")
    parser.add_argument("--output", default="data/data_preflight.json")
    parser.add_argument("--minimum-free-gib", type=float, default=80.0)
    parser.add_argument("--require-auth", action="store_true")
    args = parser.parse_args()

    report: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "cwd": str(Path.cwd()),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "HF_HOME",
                "HF_HUB_CACHE",
                "HF_XET_CACHE",
                "HF_TOKEN_PATH",
                "HF_HUB_DOWNLOAD_TIMEOUT",
                "HF_HUB_ETAG_TIMEOUT",
                "HF_XET_NUM_CONCURRENT_RANGE_GETS",
                "HF_XET_HIGH_PERFORMANCE",
            )
        },
        "packages": {},
        "checks": {},
        "errors": [],
        "warnings": [],
    }

    repo_root = Path(__file__).resolve().parents[1]
    cwd = Path.cwd().resolve()
    report["checks"]["repo_root"] = str(repo_root)
    report["checks"]["cwd_is_repo_root"] = cwd == repo_root
    if cwd != repo_root:
        report["errors"].append(
            f"Run the preflight from the repository root {repo_root}; current directory is {cwd}."
        )

    virtual_env = os.environ.get("VIRTUAL_ENV")
    report["checks"]["virtual_env"] = virtual_env
    if virtual_env:
        active_repo = Path(virtual_env).expanduser().resolve().parent
        report["checks"]["virtual_env_matches_repo"] = active_repo == repo_root
        if active_repo != repo_root:
            report["errors"].append(
                "The active virtual environment belongs to another checkout: "
                f"{Path(virtual_env).expanduser().resolve()}. Run `deactivate`, then "
                f"`source {repo_root / '.venv/bin/activate'}`."
            )
    else:
        report["checks"]["virtual_env_matches_repo"] = None
        report["warnings"].append(
            f"No virtual environment is active. Recommended: source {repo_root / '.venv/bin/activate'}"
        )

    for name in ("datasets", "huggingface_hub", "fsspec", "zstandard", "pyarrow", "yaml", "boto3"):
        package_version, error = version(name)
        report["packages"][name] = {"version": package_version, "error": error}
        if error:
            report["errors"].append(f"Cannot import {name}: {error}")

    try:
        # Import the optional codec before fsspec so its registry is populated.
        import zstandard  # noqa: F401
        from fsspec.compression import compr

        report["checks"]["fsspec_zstd_registered"] = "zstd" in compr
        if "zstd" not in compr:
            report["errors"].append(
                "fsspec did not register zstd. Run: python -m pip install --upgrade zstandard fsspec datasets"
            )
    except Exception as exc:
        report["checks"]["fsspec_zstd_registered"] = False
        report["errors"].append(f"zstd registry check failed: {type(exc).__name__}: {exc}")

    try:
        from datasets import Dataset

        iterable = Dataset.from_dict({"x": [1, 2, 3]}).to_iterable_dataset(num_shards=2)
        iterator = iter(iterable)
        next(iterator)
        cursor = iterable.state_dict()
        iterable2 = Dataset.from_dict({"x": [1, 2, 3]}).to_iterable_dataset(num_shards=2)
        iterable2.load_state_dict(cursor)
        report["checks"]["iterable_state_dict"] = True
    except Exception as exc:
        report["checks"]["iterable_state_dict"] = False
        report["errors"].append(
            f"Hugging Face shard-aware resume is unavailable: {type(exc).__name__}: {exc}. "
            "Upgrade with: python -m pip install --upgrade 'datasets>=3.5'"
        )

    try:
        from huggingface_hub import get_token

        authenticated = bool(get_token())
        report["checks"]["hf_authenticated"] = authenticated
        if not authenticated:
            message = "No Hugging Face token is configured. Run `hf auth login` for higher public-Hub rate limits."
            if args.require_auth:
                report["errors"].append(message)
            else:
                report["warnings"].append(message)
    except Exception as exc:
        report["checks"]["hf_authenticated"] = False
        report["warnings"].append(f"Could not inspect Hugging Face authentication: {exc}")

    usage = shutil.disk_usage(Path.cwd())
    free_gib = usage.free / 2**30
    report["checks"]["free_disk_gib"] = free_gib
    if free_gib < args.minimum_free_gib:
        report["errors"].append(
            f"Only {free_gib:.1f} GiB is free; at least {args.minimum_free_gib:.1f} GiB was requested."
        )

    hf_home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    try:
        hf_home.mkdir(parents=True, exist_ok=True)
        probe = hf_home / ".asterlm-write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        report["checks"]["hf_cache_writable"] = True
        report["checks"]["hf_home"] = str(hf_home)
    except Exception as exc:
        report["checks"]["hf_cache_writable"] = False
        report["errors"].append(f"HF cache is not writable at {hf_home}: {exc}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"Preflight report: {output}")
    if report["errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
