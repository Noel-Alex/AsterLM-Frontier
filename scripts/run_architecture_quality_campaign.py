#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from asterlm.artifacts import atomic_write_json
from asterlm.config import AsterConfig, DataConfig, TrainConfig
from asterlm.cuda_toolchain import require_compatible_cuda_toolchain
from asterlm.experiments import load_architecture_campaign, materialize_architecture_campaign
from asterlm.experiments.quality import (
    audit_named_initialization,
    latest_complete_checkpoint,
    summarize_quality_run,
)
from asterlm.source_provenance import assert_expected_checkout_source

DEFAULT_CANDIDATES = (
    "tier0-dense-mla-220m",
    "tier1-dense-kda3-mla-220m",
    "tier1-dense-kda3-mla-h128-samewidth-220m",
    "tier1-dense-kda3-mla-h128-expanded-reference-gate-220m",
)


def _portable(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _validate_data(data_path: Path, root: Path) -> DataConfig:
    config = DataConfig.from_yaml(data_path)
    for source in [*config.sources, *config.validation_sources]:
        candidate = Path(source.path)
        if not candidate.is_absolute():
            candidate = root / candidate
        if not candidate.exists():
            raise FileNotFoundError(f"Quality corpus source does not exist: {candidate}")
    if config.manifest_path:
        manifest = Path(config.manifest_path)
        if not manifest.is_absolute():
            manifest = root / manifest
        if not manifest.is_file():
            raise FileNotFoundError(f"Quality corpus manifest does not exist: {manifest}")
    return config


def _train_payload(
    base: dict[str, Any],
    *,
    run_dir: Path,
    seed: int,
    max_tokens: int,
    tokenizer: Path,
    train_overrides: dict[str, Any],
    no_compile: bool,
    smoke: bool,
    resume: Path | None,
) -> dict[str, Any]:
    payload = copy.deepcopy(base)
    if "train" in payload:
        train = payload["train"]
    else:
        train = payload
        payload = {"train": train}
    train.update(copy.deepcopy(train_overrides))
    if no_compile and train.get("compile") is True:
        raise ValueError(
            "--no-compile cannot relabel an execution variant that requires compile=true; "
            "select the corresponding eager variant instead"
        )
    train["output_dir"] = run_dir.as_posix()
    train["seed"] = seed
    train["deterministic_named_initialization"] = True
    train["max_tokens"] = max_tokens
    train["tokenizer_path"] = tokenizer.as_posix()
    if no_compile:
        train["compile"] = False
    tokens_per_update = (
        int(train["sequence_length"])
        * int(train["micro_batch_size"])
        * int(train["gradient_accumulation_steps"])
    )
    total_steps = math.ceil(max_tokens / tokens_per_update)
    train["warmup_steps"] = min(int(train.get("warmup_steps", 0)), max(0, total_steps - 1))
    if smoke:
        train["eval_batches"] = 1
        train["eval_interval"] = max(1, total_steps)
        train["save_interval"] = max(1, total_steps + 1)
        train["keep_last_checkpoints"] = 1
        train["milestone_tokens"] = []
        train["milestone_eval"] = False
    else:
        milestones = [
            int(value)
            for value in train.get("milestone_tokens", [])
            if 0 < int(value) < max_tokens
        ]
        train["milestone_tokens"] = sorted({*milestones, max_tokens})
    train["resume"] = resume.as_posix() if resume is not None else None
    TrainConfig(**train)
    return payload


def _execution_matrix(
    materialized: dict[str, Any],
    candidates: tuple[str, ...],
    requested_variants: tuple[str, ...],
) -> list[tuple[str, str]]:
    known_variants = set(materialized["execution_variants"])
    unknown = sorted(set(requested_variants) - known_variants)
    if unknown:
        raise ValueError(f"Unknown execution variants: {unknown}")
    requested = set(requested_variants)
    matrix: list[tuple[str, str]] = []
    for candidate_id in candidates:
        allowed = materialized["candidates"][candidate_id]["execution_variants"]
        for variant_id in allowed:
            if not requested or variant_id in requested:
                matrix.append((candidate_id, variant_id))
    if requested:
        unused = sorted(requested - {variant_id for _, variant_id in matrix})
        if unused:
            raise ValueError(
                "Requested execution variants do not apply to the selected candidates: "
                f"{unused}"
            )
    if not matrix:
        raise ValueError("The selected candidates and execution variants produce an empty matrix")
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run source-pinned, shared-initialization architecture quality comparisons"
    )
    parser.add_argument(
        "--campaign", type=Path, default=Path("configs/experiments/architecture_campaign.yaml")
    )
    parser.add_argument(
        "--train", type=Path, default=Path("configs/train/campaign_quality_2k_adamw.yaml")
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("runs/architecture-campaign/quality-data-100m-stackfree/data-proxy.yaml"),
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("artifacts/tokenizer_quality_stackfree.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("runs/architecture-campaign/quality"))
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument(
        "--execution-variant",
        action="append",
        default=[],
        help="Run only this physical execution variant; repeat to select a matched subset.",
    )
    parser.add_argument("--seed", action="append", type=int, default=[])
    parser.add_argument("--tokens", type=int, default=16_777_216)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run a storage-bounded execution gate: one evaluation batch, no periodic or "
            "permanent milestone checkpoint, and one final resumable checkpoint."
        ),
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    source = assert_expected_checkout_source(root)
    data_path = (root / args.data).resolve() if not args.data.is_absolute() else args.data.resolve()
    train_path = (root / args.train).resolve() if not args.train.is_absolute() else args.train.resolve()
    tokenizer = (
        (root / args.tokenizer).resolve()
        if not args.tokenizer.is_absolute()
        else args.tokenizer.resolve()
    )
    output = (root / args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
    campaign_path = (
        (root / args.campaign).resolve()
        if not args.campaign.is_absolute()
        else args.campaign.resolve()
    )
    if args.tokens <= 0:
        raise ValueError("--tokens must be positive")
    if not tokenizer.is_file():
        raise FileNotFoundError(f"Tokenizer does not exist: {tokenizer}")
    _validate_data(data_path, root)

    candidates = tuple(args.candidate or DEFAULT_CANDIDATES)
    seeds = tuple(args.seed or [1337])
    campaign = load_architecture_campaign(campaign_path, repo_root=root)
    materialized = materialize_architecture_campaign(campaign, output / "configs")
    unknown = sorted(set(candidates) - set(materialized["candidates"]))
    if unknown:
        raise ValueError(f"Unknown campaign candidates: {unknown}")
    execution_matrix = _execution_matrix(
        materialized,
        candidates,
        tuple(args.execution_variant),
    )
    cuda_toolchain = None
    if any(variant_id.startswith("fla-kda") for _, variant_id in execution_matrix):
        cuda_toolchain = require_compatible_cuda_toolchain()

    configs = [
        (
            candidate_id,
            AsterConfig.from_yaml(
                Path(materialized["candidates"][candidate_id]["materialized_config"])
            ),
        )
        for candidate_id in candidates
    ]
    initialization_audits = (
        {str(seed): audit_named_initialization(configs, seed) for seed in seeds}
        if len(configs) > 1
        else {
            str(seed): {
                "seed": seed,
                "reference": configs[0][0],
                "status": "single_candidate_not_applicable",
                "candidates": {},
            }
            for seed in seeds
        }
    )
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "status": "preflight_ok",
        "source_provenance": source,
        "campaign": _portable(campaign_path, root),
        "train_template": _portable(train_path, root),
        "data": _portable(data_path, root),
        "tokenizer": _portable(tokenizer, root),
        "candidates": list(candidates),
        "execution_variants": sorted({variant for _, variant in execution_matrix}),
        "execution_matrix": [
            {"candidate_id": candidate, "execution_variant": variant}
            for candidate, variant in execution_matrix
        ],
        "seeds": list(seeds),
        "tokens_per_candidate": args.tokens,
        "smoke": args.smoke,
        "initialization_audits": initialization_audits,
        "cuda_toolchain": cuda_toolchain,
        "runs": {},
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "quality-campaign.json", manifest)
    if args.preflight_only:
        print(json.dumps(manifest, indent=2))
        return

    base_train = yaml.safe_load(train_path.read_text(encoding="utf-8")) or {}
    failures = 0
    for seed in seeds:
        for candidate_id, variant_id in execution_matrix:
            run_dir = output / f"seed-{seed}" / candidate_id / variant_id
            existing = summarize_quality_run(run_dir)
            if existing["status"] == "ok" and existing["tokens_seen"] >= args.tokens:
                manifest["runs"][f"{seed}:{candidate_id}:{variant_id}"] = existing
                atomic_write_json(output / "quality-campaign.json", manifest)
                continue
            resume = latest_complete_checkpoint(run_dir)
            if (run_dir / "experiment.json").exists() and resume is None:
                raise RuntimeError(
                    f"Existing incomplete run has no resumable checkpoint: {run_dir}. "
                    "Preserve it for forensics and choose a new --output."
                )
            train_payload = _train_payload(
                base_train,
                run_dir=run_dir,
                seed=seed,
                max_tokens=args.tokens,
                tokenizer=tokenizer,
                train_overrides=materialized["execution_variants"][variant_id][
                    "train_overrides"
                ],
                no_compile=args.no_compile,
                smoke=args.smoke,
                resume=resume,
            )
            generated = (
                output
                / "train-configs"
                / f"seed-{seed}-{candidate_id}--{variant_id}.yaml"
            )
            generated.parent.mkdir(parents=True, exist_ok=True)
            generated.write_text(yaml.safe_dump(train_payload, sort_keys=False), encoding="utf-8")
            effective_variant = materialized["candidates"][candidate_id][
                "effective_variants"
            ][variant_id]
            model_path = Path(effective_variant["materialized_config"])
            command = [
                sys.executable,
                str(root / "scripts/train_pretrain.py"),
                "--model",
                str(model_path),
                "--train",
                str(generated),
                "--data",
                str(data_path),
            ]
            if resume is not None:
                command.extend(["--resume", str(resume)])
            environment = os.environ.copy()
            environment["ASTERLM_EXPERIMENT_HYPOTHESIS"] = str(
                materialized["candidates"][candidate_id]["hypothesis"]
            )
            environment["ASTERLM_EXECUTION_VARIANT"] = variant_id
            environment_delta = dict(effective_variant["environment"])
            environment.update(environment_delta)
            print("$", " ".join(command), flush=True)
            completed = subprocess.run(command, cwd=root, env=environment, check=False)
            summary = summarize_quality_run(run_dir)
            summary.update(
                {
                    "candidate_id": candidate_id,
                    "execution_variant": variant_id,
                    "seed": seed,
                    "returncode": completed.returncode,
                    "resumed_from": str(resume) if resume is not None else None,
                    "model_config": _portable(model_path, root),
                    "model_config_sha256": effective_variant["config_sha256"],
                    "train_config": _portable(generated, root),
                    "train_overrides": materialized["execution_variants"][variant_id][
                        "train_overrides"
                    ],
                    "environment": environment_delta,
                }
            )
            manifest["runs"][f"{seed}:{candidate_id}:{variant_id}"] = summary
            failures += int(completed.returncode != 0)
            manifest["status"] = "running" if failures == 0 else "partial_failure"
            atomic_write_json(output / "quality-campaign.json", manifest)
            if completed.returncode and not args.continue_on_error:
                raise SystemExit(completed.returncode)

    manifest["status"] = "complete" if failures == 0 else "partial_failure"
    atomic_write_json(output / "quality-campaign.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
