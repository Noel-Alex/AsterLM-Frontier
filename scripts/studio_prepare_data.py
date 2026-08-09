#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], *, dry_run: bool = False) -> None:
    print("$", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def normalize_weights(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total = sum(max(0.0, float(item.get("weight", 0.0))) for item in items)
    if total <= 0:
        equal = 1.0 / max(1, len(items))
        return [{**item, "weight": equal} for item in items]
    return [{**item, "weight": float(item.get("weight", 0.0)) / total} for item in items]


def directory_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean/decontaminate an arbitrary Aster Studio corpus plan and generate a training DataConfig."
    )
    parser.add_argument("--plan", required=True)
    parser.add_argument("--reset-existing", action="store_true")
    parser.add_argument(
        "--allow-no-benchmarks",
        action="store_true",
        help="Explicitly allow cleaning without benchmark decontamination. Studio UI does not enable this.",
    )
    parser.add_argument(
        "--allow-low-disk",
        action="store_true",
        help="Override Studio's conservative clean-copy disk preflight.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    plan_path = Path(args.plan)
    if not plan_path.is_absolute():
        plan_path = ROOT / plan_path
    raw = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    plan = raw.get("plan", raw)

    output = ROOT / str(plan.get("output", "data/clean-frontier-studio"))
    benchmarks = ROOT / str(plan.get("benchmarks", "data/decontamination-benchmarks"))
    validation_fraction = float(plan.get("validation_fraction", 0.005))
    pii_mode = str(plan.get("pii_mode", "redact"))
    near_distance = int(plan.get("near_distance", 3))
    audit_sample = int(plan.get("audit_sample", 10000))
    sources = list(plan.get("sources", []))
    if not sources:
        raise SystemExit("Plan has no sources.")
    if not 0.0 < validation_fraction < 1.0:
        raise SystemExit("validation_fraction must be in (0,1)")

    raw_paths = []
    for item in sources:
        raw_path = ROOT / str(item["raw_path"])
        if raw_path.exists():
            raw_paths.append(raw_path)
    raw_bytes = sum(directory_bytes(path) for path in raw_paths)
    free_bytes = shutil.disk_usage(ROOT).free
    # Cleaning writes a second compressed corpus and validation holdouts. Exact
    # compression is source dependent; reserve roughly one raw-copy plus 20 GiB.
    conservative_need = int(raw_bytes * 1.05 + 20 * 2**30)
    if (
        raw_bytes > 0
        and free_bytes < conservative_need
        and not args.allow_low_disk
        and not args.dry_run
    ):
        raise RuntimeError(
            "Insufficient free disk for a conservative clean-copy estimate: "
            f"raw={raw_bytes / 2**30:.1f} GiB, free={free_bytes / 2**30:.1f} GiB, "
            f"recommended>={conservative_need / 2**30:.1f} GiB. "
            "Prune only reconstructable caches/add storage, or explicitly pass "
            "--allow-low-disk after reviewing the risk."
        )

    if args.reset_existing and output.exists() and not args.dry_run:
        import shutil
        shutil.rmtree(output)

    output.mkdir(parents=True, exist_ok=True)
    benchmark_arg = ["--benchmark", str(benchmarks)] if benchmarks.exists() else []
    if not benchmark_arg and not args.allow_no_benchmarks:
        raise FileNotFoundError(
            f"Benchmark/decontamination directory is missing: {benchmarks}. "
            "Download/verify the benchmark profile first. Use --allow-no-benchmarks only "
            "when intentionally creating a non-decontaminated experimental derivative."
        )
    if not benchmark_arg:
        print(
            "WARNING: --allow-no-benchmarks was explicitly supplied; "
            "the clean derivative will NOT be benchmark-decontaminated."
        )

    # One shared deduplication index is used across every selected source.
    # This catches FineWeb↔DCLM↔math↔code duplicates rather than deduplicating
    # each source in isolation.
    global_command = [
        sys.executable,
        "scripts/studio_clean_corpus.py",
        "--plan",
        str(plan_path),
        "--checkpoint-records",
        str(int(plan.get("clean_checkpoint_records", 100000))),
        "--shard-mb",
        str(int(plan.get("clean_shard_mb", 512))),
    ]
    if args.allow_no_benchmarks:
        global_command.append("--allow-no-benchmarks")
    run(global_command, dry_run=args.dry_run)

    clean_sources: list[dict[str, Any]] = []
    for item in sources:
        sid = str(item["id"])
        raw_path = ROOT / str(item["raw_path"])
        if not raw_path.exists():
            if bool(item.get("optional", False)):
                print(f"Skipping optional missing source: {sid} -> {raw_path}")
                continue
            raise FileNotFoundError(f"Missing raw source {sid}: {raw_path}")

        destination = output / sid
        validation_destination = output / "validation" / sid
        run(
            [
                sys.executable,
                "scripts/audit_corpus.py",
                "--input",
                str(destination),
                "--sample",
                str(audit_sample),
            ],
            dry_run=args.dry_run,
        )

        clean_sources.append(
            {
                "id": sid,
                "path": str(destination.relative_to(ROOT)),
                "validation_path": str(validation_destination.relative_to(ROOT)),
                "text_field": "text",
                "weight": float(item.get("weight", 0.0)),
                "fim_rate": float(item.get("fim_rate", 0.0)),
            }
        )

    if not clean_sources:
        raise SystemExit("No sources remained after applying the plan.")

    clean_sources = normalize_weights(clean_sources)
    data_sources: list[dict[str, Any]] = []
    validation_sources: list[dict[str, Any]] = []
    for item in clean_sources:
        base = {
            "path": item["path"],
            "text_field": item["text_field"],
            "weight": item["weight"],
        }
        if item["fim_rate"] > 0:
            base["fim_rate"] = item["fim_rate"]
        data_sources.append(base)

        val = {
            "path": item["validation_path"],
            "text_field": item["text_field"],
            "weight": item["weight"],
        }
        if item["fim_rate"] > 0:
            val["fim_rate"] = 0.0
        validation_sources.append(val)

    generated_data = {
        "data": {
            "seed": int(plan.get("seed", 1337)),
            "shuffle_buffer": int(plan.get("shuffle_buffer", 20000)),
            "min_chars": int(plan.get("min_chars", 128)),
            "max_chars": int(plan.get("max_chars", 500000)),
            "quality_filters": bool(plan.get("quality_filters", True)),
            "add_eos_between_documents": True,
            "mask_cross_document_loss": True,
            "sources": data_sources,
            "validation_sources": validation_sources,
        }
    }

    config_name = str(plan.get("generated_config", f"configs/studio/data/{plan_path.stem}_clean.yaml"))
    config_path = ROOT / config_name
    summary_path = output / "studio_prepare_summary.json"
    summary = {
        "version": 1,
        "plan": str(plan_path),
        "output": str(output),
        "generated_config": str(config_path),
        "decontaminated": bool(benchmark_arg),
        "validation_fraction": validation_fraction,
        "sources": clean_sources,
    }

    if not args.dry_run:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(generated_data, sort_keys=False), encoding="utf-8")
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
