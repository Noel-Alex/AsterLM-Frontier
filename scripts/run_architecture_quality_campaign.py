#!/usr/bin/env python3
from __future__ import annotations

import argparse
import atexit
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
from asterlm.cuda_allocator import cuda_allocator_environment
from asterlm.cuda_toolchain import require_compatible_cuda_toolchain
from asterlm.experiments import load_architecture_campaign, materialize_architecture_campaign
from asterlm.experiments.quality import (
    archive_incomplete_quality_run,
    audit_named_initialization,
    latest_complete_checkpoint,
    summarize_quality_run,
)
from asterlm.experiments.quality_analysis import analyze_quality_campaign
from asterlm.experiments.source_checkout import create_pinned_source_checkout
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


def _refresh_analysis(campaign_path: Path) -> None:
    atomic_write_json(
        campaign_path.with_name("quality-analysis.json"),
        analyze_quality_campaign(campaign_path),
    )


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


def _absolute_data_config(data_path: Path, root: Path, target: Path) -> Path:
    payload = yaml.safe_load(data_path.read_text(encoding="utf-8")) or {}
    data = payload.get("data", payload)
    for key in ("sources", "validation_sources"):
        for source in data.get(key, []):
            candidate = Path(str(source["path"]))
            if not candidate.is_absolute() and (root / candidate).exists():
                source["path"] = str((root / candidate).resolve())
    manifest = data.get("manifest_path")
    if manifest:
        candidate = Path(str(manifest))
        if not candidate.is_absolute() and (root / candidate).is_file():
            data["manifest_path"] = str((root / candidate).resolve())
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return target


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
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    payload = copy.deepcopy(base)
    if "train" in payload:
        train = payload["train"]
    else:
        train = payload
        payload = {"train": train}
    train.update(copy.deepcopy(train_overrides))
    if environment and environment.get("ASTER_MOE_IMPL"):
        train["moe_implementation"] = environment["ASTER_MOE_IMPL"]
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


def _explicit_execution_matrix(
    materialized: dict[str, Any], requested_runs: tuple[str, ...]
) -> tuple[tuple[str, ...], list[tuple[str, str]]]:
    matrix: list[tuple[str, str]] = []
    candidates: list[str] = []
    known_candidates = set(materialized["candidates"])
    for raw in requested_runs:
        if "=" not in raw:
            raise ValueError(f"Invalid --run {raw!r}; expected CANDIDATE=EXECUTION_VARIANT")
        candidate_id, variant_id = (part.strip() for part in raw.split("=", 1))
        if candidate_id not in known_candidates:
            raise ValueError(f"Unknown campaign candidate in --run: {candidate_id}")
        allowed = materialized["candidates"][candidate_id]["execution_variants"]
        if variant_id not in allowed:
            raise ValueError(
                f"Execution variant {variant_id!r} is not allowed for {candidate_id}; "
                f"choose one of {allowed}"
            )
        pair = (candidate_id, variant_id)
        if pair in matrix:
            raise ValueError(f"Duplicate --run pair: {raw}")
        matrix.append(pair)
        if candidate_id not in candidates:
            candidates.append(candidate_id)
    if not matrix:
        raise ValueError("At least one CANDIDATE=EXECUTION_VARIANT pair is required")
    return tuple(candidates), matrix


def _validate_resume_contract(
    existing: dict[str, Any],
    *,
    candidates: tuple[str, ...],
    execution_matrix: list[tuple[str, str]],
    seeds: tuple[int, ...],
    tokens: int,
    smoke: bool,
) -> None:
    """Refuse to blend results from different campaign contracts."""

    expected_matrix = [
        {"candidate_id": candidate, "execution_variant": variant}
        for candidate, variant in execution_matrix
    ]
    checks = {
        "candidates": (existing.get("candidates"), list(candidates)),
        "execution_matrix": (existing.get("execution_matrix"), expected_matrix),
        "seeds": (existing.get("seeds"), list(seeds)),
        "tokens_per_candidate": (existing.get("tokens_per_candidate"), tokens),
        "smoke": (bool(existing.get("smoke", False)), smoke),
    }
    mismatches = {
        name: {"existing": observed, "requested": requested}
        for name, (observed, requested) in checks.items()
        if observed != requested
    }
    if mismatches:
        raise ValueError(
            "--resume-existing must use the original campaign contract; "
            f"mismatches={mismatches}"
        )
    original_commit = (existing.get("source_provenance") or {}).get("git_commit")
    if not original_commit:
        raise ValueError("Existing campaign has no source-pinned git commit")


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
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="CANDIDATE=EXECUTION_VARIANT",
        help=(
            "Select an exact architecture/execution pair; repeat for a heterogeneous "
            "matched matrix such as dense BF16 versus sparse FP8. Cannot be combined "
            "with --candidate or --execution-variant."
        ),
    )
    parser.add_argument("--seed", action="append", type=int, default=[])
    parser.add_argument("--tokens", type=int, default=16_777_216)
    parser.add_argument(
        "--checkpoint-policy",
        choices=("none", "final_only", "full"),
        default="none",
        help=(
            "Checkpoint retention for candidate training. The default keeps complete "
            "metrics/evaluations but no multi-gigabyte model states."
        ),
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run a storage-bounded execution gate: one evaluation batch, no periodic or "
            "permanent milestone checkpoint. A final checkpoint is written only when "
            "--checkpoint-policy is final_only or full."
        ),
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help=(
            "Resume this exact output contract. Completed runs are retained, an "
            "interrupted metrics-only attempt is archived, and new work executes "
            "from the campaign's original source commit."
        ),
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    source = assert_expected_checkout_source(root)
    if source.get("dirty"):
        raise RuntimeError(
            "Architecture quality campaigns require a clean checkout before source pinning"
        )
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

    seeds = tuple(args.seed or [1337])
    existing_manifest_path = output / "quality-campaign.json"
    existing_manifest: dict[str, Any] | None = None
    if args.resume_existing:
        if not existing_manifest_path.is_file():
            raise FileNotFoundError(
                f"--resume-existing requires {existing_manifest_path}"
            )
        existing_manifest = json.loads(existing_manifest_path.read_text(encoding="utf-8"))
        materialized_path = output / "configs" / "campaign-manifest.json"
        if not materialized_path.is_file():
            raise FileNotFoundError(
                "--resume-existing requires the source-pinned materialized campaign "
                f"manifest: {materialized_path}"
            )
        materialized = json.loads(materialized_path.read_text(encoding="utf-8"))
    else:
        campaign = load_architecture_campaign(campaign_path, repo_root=root)
        materialized = materialize_architecture_campaign(campaign, output / "configs")
    if args.run:
        if args.candidate or args.execution_variant:
            raise ValueError("--run cannot be combined with --candidate or --execution-variant")
        candidates, execution_matrix = _explicit_execution_matrix(
            materialized, tuple(args.run)
        )
    else:
        candidates = tuple(args.candidate or DEFAULT_CANDIDATES)
        unknown = sorted(set(candidates) - set(materialized["candidates"]))
        if unknown:
            raise ValueError(f"Unknown campaign candidates: {unknown}")
        execution_matrix = _execution_matrix(
            materialized,
            candidates,
            tuple(args.execution_variant),
        )
    configs = [
        (
            candidate_id,
            AsterConfig.from_yaml(
                Path(materialized["candidates"][candidate_id]["materialized_config"])
            ),
        )
        for candidate_id in candidates
    ]
    cuda_toolchain = None
    if any(
        "kda" in config.pattern and config.kda_backend in {"auto", "fla"}
        for _, config in configs
    ):
        cuda_toolchain = require_compatible_cuda_toolchain()
    if existing_manifest is not None:
        _validate_resume_contract(
            existing_manifest,
            candidates=candidates,
            execution_matrix=execution_matrix,
            seeds=seeds,
            tokens=args.tokens,
            smoke=args.smoke,
        )
        manifest = copy.deepcopy(existing_manifest)
        manifest["status"] = "preflight_ok"
        manifest["resume_orchestrator_provenance"] = source
        manifest["cuda_toolchain_resume_check"] = cuda_toolchain
        manifest.setdefault("interrupted_attempts", [])
        manifest.setdefault("runs", {})
        execution_commit = str(manifest["source_provenance"]["git_commit"])
    else:
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
        manifest = {
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
            "interrupted_attempts": [],
            "runs": {},
        }
        execution_commit = str(source["git_commit"])
    output.mkdir(parents=True, exist_ok=True)
    execution_data_path = _absolute_data_config(
        data_path, root, output / "execution-data-absolute.yaml"
    )
    atomic_write_json(output / "quality-campaign.json", manifest)
    _refresh_analysis(output / "quality-campaign.json")
    if args.preflight_only:
        print(json.dumps(manifest, indent=2))
        return

    base_train = yaml.safe_load(train_path.read_text(encoding="utf-8")) or {}
    base_train_section = base_train.get("train", base_train)
    base_train_section["checkpoint_policy"] = args.checkpoint_policy
    pinned = create_pinned_source_checkout(root, execution_commit)
    atexit.register(pinned.close)
    manifest["execution_checkout"] = pinned.manifest()
    manifest["execution_data"] = str(execution_data_path)
    atomic_write_json(output / "quality-campaign.json", manifest)
    _refresh_analysis(output / "quality-campaign.json")
    failures = 0
    for seed in seeds:
        for candidate_id, variant_id in execution_matrix:
            run_dir = output / f"seed-{seed}" / candidate_id / variant_id
            existing = summarize_quality_run(run_dir)
            if existing["status"] == "ok" and existing["tokens_seen"] >= args.tokens:
                key = f"{seed}:{candidate_id}:{variant_id}"
                previous = manifest["runs"].get(key, {})
                manifest["runs"][key] = {**previous, **existing}
                atomic_write_json(output / "quality-campaign.json", manifest)
                _refresh_analysis(output / "quality-campaign.json")
                continue
            resume = latest_complete_checkpoint(run_dir)
            if (run_dir / "experiment.json").exists() and resume is None:
                if args.checkpoint_policy != "none":
                    raise RuntimeError(
                        f"Existing incomplete run has no resumable checkpoint: {run_dir}. "
                        "A checkpoint-retaining campaign must fail closed rather than "
                        "discard recoverability evidence."
                    )
                interrupted_summary = summarize_quality_run(run_dir)
                archive = archive_incomplete_quality_run(
                    run_dir,
                    output / "interrupted-attempts" / f"seed-{seed}" / candidate_id,
                )
                manifest["interrupted_attempts"].append(
                    {
                        "seed": seed,
                        "candidate_id": candidate_id,
                        "execution_variant": variant_id,
                        "reason": "interrupted_metrics_only_run_without_checkpoint",
                        "archive": _portable(archive, root),
                        "summary": interrupted_summary,
                    }
                )
                atomic_write_json(output / "quality-campaign.json", manifest)
                _refresh_analysis(output / "quality-campaign.json")
            train_payload = _train_payload(
                base_train,
                run_dir=run_dir,
                seed=seed,
                max_tokens=args.tokens,
                tokenizer=tokenizer,
                train_overrides=materialized["execution_variants"][variant_id][
                    "train_overrides"
                ],
                environment=materialized["execution_variants"][variant_id]["environment"],
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
                str(pinned.path / "scripts/train_pretrain.py"),
                "--model",
                str(model_path),
                "--train",
                str(generated),
                "--data",
                str(execution_data_path),
            ]
            if resume is not None:
                command.extend(["--resume", str(resume)])
            environment = dict(os.environ)
            pinned_pythonpath = str(pinned.path / "src")
            existing_pythonpath = environment.get("PYTHONPATH")
            environment["PYTHONPATH"] = (
                pinned_pythonpath
                if not existing_pythonpath
                else os.pathsep.join((pinned_pythonpath, existing_pythonpath))
            )
            environment["ASTERLM_EXPERIMENT_HYPOTHESIS"] = str(
                materialized["candidates"][candidate_id]["hypothesis"]
            )
            environment["ASTERLM_EXECUTION_VARIANT"] = variant_id
            environment_delta = dict(effective_variant["environment"])
            environment.update(environment_delta)
            environment = cuda_allocator_environment(environment)
            print("$", " ".join(command), flush=True)
            completed = subprocess.run(
                command, cwd=pinned.path, env=environment, check=False
            )
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
                    "comparison_role": materialized["execution_variants"][variant_id][
                        "comparison_role"
                    ],
                    "numerical_family": materialized["execution_variants"][variant_id][
                        "numerical_family"
                    ],
                    "environment": environment_delta,
                }
            )
            manifest["runs"][f"{seed}:{candidate_id}:{variant_id}"] = summary
            failures += int(completed.returncode != 0)
            manifest["status"] = "running" if failures == 0 else "partial_failure"
            atomic_write_json(output / "quality-campaign.json", manifest)
            _refresh_analysis(output / "quality-campaign.json")
            if completed.returncode and not args.continue_on_error:
                raise SystemExit(completed.returncode)

    manifest["status"] = "complete" if failures == 0 else "partial_failure"
    atomic_write_json(output / "quality-campaign.json", manifest)
    _refresh_analysis(output / "quality-campaign.json")
    pinned.close()
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
