#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from asterlm.data.clean_manifest import build_clean_corpus_manifest

SOURCES = ["fineweb_edu", "dclm", "cosmopedia_v2", "finemath_4plus"]
BASE_WEIGHTS = {
    "fineweb_edu": 0.53,
    "dclm": 0.11,
    "cosmopedia_v2": 0.09,
    "finemath_4plus": 0.13,
}


def parse_assignments(values: list[str], *, value_type: type = str) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for value in values:
        key, separator, raw = value.partition("=")
        key = key.strip()
        raw = raw.strip()
        if not separator or not key or not raw:
            raise ValueError(f"Expected ID=VALUE, got {value!r}")
        if key in parsed:
            raise ValueError(f"Duplicate assignment for {key!r}")
        parsed[key] = value_type(raw)
    return parsed


def normalized_source_specs(
    jobs: list[tuple[str, Path, str]],
    *,
    weights: dict[str, float],
    fim_sources: set[str],
) -> list[dict[str, object]]:
    ids = [name for name, _, _ in jobs]
    if len(ids) != len(set(ids)):
        raise ValueError("Clean source ids must be unique")
    unknown_weights = set(weights) - set(ids)
    unknown_fim = fim_sources - set(ids)
    if unknown_weights or unknown_fim:
        raise ValueError(
            f"Source policy names unknown ids: weights={sorted(unknown_weights)}, fim={sorted(unknown_fim)}"
        )
    missing_weights = set(ids) - set(weights)
    if missing_weights:
        raise ValueError(f"Missing source weights: {sorted(missing_weights)}")
    if any(weight <= 0 for weight in weights.values()):
        raise ValueError("All source weights must be positive")
    total = sum(weights[name] for name in ids)
    return [
        {
            "id": name,
            "path": str(path),
            "text_field": field,
            "weight": weights[name] / total,
            "fim_rate": 0.5 if name in fim_sources else 0.0,
        }
        for name, path, field in jobs
    ]


def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean, deduplicate, redact, and decontaminate the materialized frontier corpus"
    )
    parser.add_argument("--raw-corpus", "--corpus-dir", dest="raw_corpus", default="data/corpus-frontier-16b")
    parser.add_argument(
        "--raw-code",
        dest="raw_code",
        default=None,
        help="Optional separately audited code-corpus directory; Stack-Edu is retired",
    )
    parser.add_argument("--code-id", default="code", help="Source id for --raw-code")
    parser.add_argument(
        "--extra-source",
        action="append",
        default=[],
        metavar="ID=PATH",
        help="Add a separately labeled supplementary source; repeat as needed",
    )
    parser.add_argument(
        "--source-weight",
        action="append",
        default=[],
        metavar="ID=FLOAT",
        help="Override/add a pre-normalization mixture weight",
    )
    parser.add_argument(
        "--fim-source",
        action="append",
        default=[],
        metavar="ID",
        help="Enable 50 percent fill-in-the-middle augmentation only for this code source",
    )
    parser.add_argument(
        "--benchmarks", "--benchmark-dir", dest="benchmarks", default="data/decontamination-benchmarks"
    )
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
    shared_dedup_db = output / "_dedup" / "global.sqlite"
    benchmark_arg = ["--benchmark", args.benchmarks] if Path(args.benchmarks).exists() else []
    if not benchmark_arg:
        print("WARNING: benchmark directory is missing; cleaning will run without decontamination")

    jobs = [(name, Path(args.raw_corpus) / name, "text") for name in SOURCES]
    extra_sources = parse_assignments(args.extra_source)
    jobs.extend((name, Path(str(path)), "text") for name, path in extra_sources.items())
    weights = dict(BASE_WEIGHTS)
    weights.update(
        {
            name: float(value)
            for name, value in parse_assignments(args.source_weight, value_type=float).items()
        }
    )
    fim_sources = set(args.fim_source)
    if args.raw_code and not args.skip_code:
        if args.code_id in extra_sources:
            raise ValueError(f"--code-id duplicates --extra-source id {args.code_id!r}")
        jobs.append((args.code_id, Path(args.raw_code), "text"))
        weights.setdefault(args.code_id, 0.14)
        fim_sources.add(args.code_id)
    specs = normalized_source_specs(jobs, weights=weights, fim_sources=fim_sources)
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
                "--dedup-db",
                str(shared_dedup_db),
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
            "manifest_path": str(output / "clean_manifest.json"),
            "validation_role": "architecture_holdout",
            "sources": [],
            "validation_sources": [],
        }
    }
    for spec in specs:
        source_id = str(spec["id"])
        config["data"]["sources"].append(
            {
                "path": str(output / source_id),
                "text_field": spec["text_field"],
                "weight": spec["weight"],
                "fim_rate": spec["fim_rate"],
            }
        )
        config["data"]["validation_sources"].append(
            {
                "path": str(output / "validation" / source_id),
                "text_field": spec["text_field"],
                "weight": spec["weight"],
                "fim_rate": 0.0,
            }
        )
    config_path = output / "pretrain_data.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    summary = {
        "format_version": 3,
        "clean_root": str(output),
        "generated_config": str(config_path),
        "decontaminated": bool(benchmark_arg),
        "validation_fraction": args.validation_fraction,
        "validation_root": str(output / "validation"),
        "sources": [name for name, _, _ in jobs],
        "source_policy": specs,
        "shared_dedup_db": str(shared_dedup_db),
    }
    (output / "prepare_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest = build_clean_corpus_manifest(
        data_config_path=config_path,
        output_path=output / "clean_manifest.json",
        benchmark_decontaminated=bool(benchmark_arg),
        pii_handled=args.pii_mode != "keep",
    )
    summary["clean_manifest"] = str(output / "clean_manifest.json")
    summary["manifest_artifacts"] = len(manifest["artifacts"])
    (output / "prepare_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
