#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


SOURCES = ["fineweb_edu", "dclm", "cosmopedia_v2", "finemath_4plus"]


def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean, deduplicate, redact, and decontaminate the materialized frontier corpus"
    )
    parser.add_argument("--raw-corpus", "--corpus-dir", dest="raw_corpus", default="data/corpus-frontier-16b")
    parser.add_argument("--raw-code", "--stack-dir", dest="raw_code", default="data/stack-edu-frontier-2p4b")
    parser.add_argument("--benchmarks", "--benchmark-dir", dest="benchmarks", default="data/decontamination-benchmarks")
    parser.add_argument("--output", "--output-dir", dest="output", default="data/clean-frontier")
    parser.add_argument("--pii-mode", choices=["redact", "drop", "keep"], default="redact")
    parser.add_argument("--near-distance", type=int, default=3)
    parser.add_argument("--skip-code", action="store_true")
    parser.add_argument("--audit-sample", type=int, default=10000)
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.005,
        help="Deterministic per-source holdout fraction routed out of training",
    )
    parser.add_argument(
        "--reset-existing",
        action="store_true",
        help="Delete an existing clean output before rebuilding it with the current policy",
    )
    args = parser.parse_args()
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be in (0, 1)")

    output = Path(args.output)
    if args.reset_existing and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    benchmark_arg = ["--benchmark", args.benchmarks] if Path(args.benchmarks).exists() else []
    if not benchmark_arg:
        print("WARNING: benchmark directory is missing; cleaning will run without decontamination")

    jobs = [(name, Path(args.raw_corpus) / name, "text") for name in SOURCES]
    if not args.skip_code:
        jobs.append(("stack_edu", Path(args.raw_code), "text"))
    for name, source, field in jobs:
        if not source.exists():
            raise FileNotFoundError(f"Missing {source}; run scripts/download_data.py first")
        destination = output / name
        validation_destination = output / "validation" / name
        if (
            not args.reset_existing
            and any(destination.glob("clean-*.jsonl.zst"))
            and not any(validation_destination.glob("clean-*.jsonl.zst"))
        ):
            raise RuntimeError(
                f"{destination} was produced without the current disjoint validation split. "
                "Rerun with --reset-existing to rebuild clean data safely."
            )
        run(
            [
                sys.executable,
                "scripts/clean_corpus.py",
                "--input",
                str(source),
                "--output",
                str(destination),
                "--validation-output",
                str(validation_destination),
                "--validation-fraction",
                str(args.validation_fraction),
                "--text-field",
                field,
                "--source-id",
                name,
                "--pii-mode",
                args.pii_mode,
                "--near-distance",
                str(args.near_distance),
                *benchmark_arg,
            ]
        )
        run(
            [
                sys.executable,
                "scripts/audit_corpus.py",
                "--input",
                str(destination),
                "--sample",
                str(args.audit_sample),
            ]
        )

    config = {
        "data": {
            "seed": 1337,
            "shuffle_buffer": 20000,
            "min_chars": 128,
            "max_chars": 500000,
            "quality_filters": True,
            "add_eos_between_documents": True,
            "mask_cross_document_loss": True,
            "sources": [
                {"path": str(output / "fineweb_edu"), "text_field": "text", "weight": 0.53},
                {"path": str(output / "dclm"), "text_field": "text", "weight": 0.11},
                {"path": str(output / "cosmopedia_v2"), "text_field": "text", "weight": 0.09},
                {"path": str(output / "finemath_4plus"), "text_field": "text", "weight": 0.13},
                {
                    "path": str(output / "stack_edu"),
                    "text_field": "text",
                    "weight": 0.14,
                    "fim_rate": 0.5,
                },
            ],
            "validation_sources": [
                {"path": str(output / "validation" / "fineweb_edu"), "text_field": "text", "weight": 0.53},
                {"path": str(output / "validation" / "dclm"), "text_field": "text", "weight": 0.11},
                {"path": str(output / "validation" / "cosmopedia_v2"), "text_field": "text", "weight": 0.09},
                {"path": str(output / "validation" / "finemath_4plus"), "text_field": "text", "weight": 0.13},
                {
                    "path": str(output / "validation" / "stack_edu"),
                    "text_field": "text",
                    "weight": 0.14,
                    "fim_rate": 0.0,
                },
            ],
        }
    }
    if args.skip_code:
        config["data"]["sources"] = [
            source for source in config["data"]["sources"] if "stack_edu" not in source["path"]
        ]
        config["data"]["validation_sources"] = [
            source
            for source in config["data"]["validation_sources"]
            if "stack_edu" not in source["path"]
        ]
        for key in ("sources", "validation_sources"):
            total = sum(source["weight"] for source in config["data"][key])
            for source in config["data"][key]:
                source["weight"] /= total
    config_path = Path("configs/data/pretrain_frontier_clean.yaml")
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    summary = {
        "format_version": 2,
        "clean_root": str(output),
        "generated_config": str(config_path),
        "decontaminated": bool(benchmark_arg),
        "validation_fraction": args.validation_fraction,
        "validation_root": str(output / "validation"),
        "sources": [name for name, _, _ in jobs],
    }
    (output / "prepare_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
