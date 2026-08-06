#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


TIERS = {
    "18b": {"pretrain": "frontier", "full": "all", "raw_tokens": 18_400_000_000},
    "50b": {"pretrain": "overtrain50", "full": "campaign50", "raw_tokens": 50_000_000_000},
    "100b": {"pretrain": "overtrain100", "full": "campaign100", "raw_tokens": 100_000_000_000},
}


def run(command: list[str], *, dry_run: bool = False) -> None:
    print("$", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one resumable AsterLM data campaign from validation through optional cleaning"
    )
    parser.add_argument("--tier", choices=sorted(TIERS), default="100b")
    parser.add_argument("--pretraining-only", action="store_true")
    parser.add_argument("--network-mode", choices=["low", "balanced", "fast"], default="low")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--reset-clean", action="store_true")
    parser.add_argument("--prune-hf-cache-before-clean", action="store_true")
    parser.add_argument("--verify-last-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    if args.prune_hf_cache_before_clean and not args.clean:
        raise SystemExit("--prune-hf-cache-before-clean requires --clean")

    tier = TIERS[args.tier]
    profile = tier["pretrain"] if args.pretraining_only else tier["full"]
    started = time.time()
    run(
        [
            sys.executable,
            "scripts/download_data.py",
            "--profile",
            profile,
            "--validate-first",
            "--require-auth",
            "--network-mode",
            args.network_mode,
            "--max-retries",
            "0",
        ],
        dry_run=args.dry_run,
    )

    verify_roots = [
        Path("data/corpus-frontier-16b"),
        Path("data/stack-edu-frontier-2p4b"),
    ]
    if not args.pretraining_only:
        verify_roots.extend(
            [
                Path("data/decontamination-benchmarks"),
                Path("data/posttrain-frontier"),
                Path("data/reasoning-frontier"),
            ]
        )
    for root in verify_roots:
        command = [sys.executable, "scripts/verify_data_shards.py", str(root)]
        if args.verify_last_only:
            command.append("--only-last")
        run(command, dry_run=args.dry_run)

    if args.clean:
        cache = Path("data/hf-cache")
        if args.prune_hf_cache_before_clean and cache.exists() and not args.dry_run:
            print(f"Removing reconstructable Hugging Face cache before cleaning: {cache}")
            shutil.rmtree(cache)
        command = [
            sys.executable,
            "scripts/prepare_frontier_data.py",
            "--raw-corpus",
            "data/corpus-frontier-16b",
            "--raw-code",
            "data/stack-edu-frontier-2p4b",
            "--benchmarks",
            "data/decontamination-benchmarks",
            "--output",
            "data/clean-frontier",
        ]
        if args.reset_clean:
            command.append("--reset-existing")
        run(command, dry_run=args.dry_run)

    report_path = Path(args.report or f"data/campaign-{args.tier}.json")
    report = {
        "version": 1,
        "tier": args.tier,
        "profile": profile,
        "target_raw_tokens": tier["raw_tokens"],
        "pretraining_only": args.pretraining_only,
        "clean_requested": args.clean,
        "dry_run": args.dry_run,
        "started_at_unix": started,
        "finished_at_unix": time.time(),
        "paths": {
            str(path): {
                "exists": path.exists(),
                "bytes": directory_size(path),
            }
            for path in [*verify_roots, Path("data/clean-frontier")]
        },
    }
    if not args.dry_run:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
