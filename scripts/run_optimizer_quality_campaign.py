#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from asterlm.artifacts import atomic_write_json
from asterlm.config import AsterConfig, TrainConfig
from asterlm.cuda_toolchain import require_compatible_cuda_toolchain
from asterlm.experiments.quality import (
    archive_incomplete_quality_run,
    summarize_quality_run,
)
from asterlm.experiments.quality_analysis import analyze_quality_campaign
from asterlm.experiments.source_checkout import create_pinned_source_checkout
from asterlm.source_provenance import assert_expected_checkout_source
from scripts.run_architecture_quality_campaign import (
    _absolute_data_config,
    _portable,
    _train_payload,
    _validate_data,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_optimizer_campaign(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if payload.get("schema_version") != 1:
        raise ValueError("optimizer campaign schema_version must be 1")
    arms = payload.get("arms")
    if not isinstance(arms, dict) or not arms:
        raise ValueError("optimizer campaign requires at least one arm")
    for arm_id, arm in arms.items():
        if not isinstance(arm, dict) or not isinstance(arm.get("train_overrides"), dict):
            raise TypeError(f"optimizer arm {arm_id!r} requires train_overrides")
        warmup = float(arm.get("warmup_fraction", 0.0))
        if not 0.0 <= warmup < 1.0:
            raise ValueError(f"optimizer arm {arm_id!r} warmup_fraction must be in [0, 1)")
        if not str(arm.get("family", "")).strip():
            raise ValueError(f"optimizer arm {arm_id!r} requires a family")
    return payload


def _refresh_analysis(path: Path) -> None:
    atomic_write_json(path.with_name("quality-analysis.json"), analyze_quality_campaign(path))


def _validate_resume_manifest(
    existing: dict[str, Any],
    *,
    arm_ids: tuple[str, ...],
    arms: dict[str, Any],
    seeds: tuple[int, ...],
    tokens: int,
    checkpoint_policy: str,
) -> None:
    checks = {
        "execution_variants": (existing.get("execution_variants"), list(arm_ids)),
        "arms": (existing.get("arms"), arms),
        "seeds": (existing.get("seeds"), list(seeds)),
        "tokens_per_candidate": (existing.get("tokens_per_candidate"), tokens),
        "checkpoint_policy": (
            existing.get("checkpoint_policy"),
            checkpoint_policy,
        ),
    }
    mismatches = {
        name: {"existing": observed, "requested": requested}
        for name, (observed, requested) in checks.items()
        if observed != requested
    }
    if mismatches:
        raise ValueError(f"Optimizer resume contract mismatch: {mismatches}")
    if not (existing.get("source_provenance") or {}).get("git_commit"):
        raise ValueError("Optimizer resume manifest has no source-pinned commit")


def _arm_train_payload(
    base: dict[str, Any],
    *,
    run_dir: Path,
    seed: int,
    max_tokens: int,
    tokenizer: Path,
    common_overrides: dict[str, Any],
    arm: dict[str, Any],
    checkpoint_policy: str,
) -> dict[str, Any]:
    overrides = {**copy.deepcopy(common_overrides), **copy.deepcopy(arm["train_overrides"])}
    overrides["checkpoint_policy"] = checkpoint_policy
    section = base.get("train", base)
    sequence = int(overrides.get("sequence_length", section["sequence_length"]))
    micro_batch = int(overrides.get("micro_batch_size", section["micro_batch_size"]))
    accumulation = int(
        overrides.get(
            "gradient_accumulation_steps", section["gradient_accumulation_steps"]
        )
    )
    total_steps = math.ceil(max_tokens / (sequence * micro_batch * accumulation))
    warmup_fraction = float(arm.get("warmup_fraction", 0.0))
    overrides["warmup_steps"] = min(
        max(0, math.ceil(total_steps * warmup_fraction)),
        max(total_steps - 1, 0),
    )
    return _train_payload(
        base,
        run_dir=run_dir,
        seed=seed,
        max_tokens=max_tokens,
        tokenizer=tokenizer,
        train_overrides=overrides,
        no_compile=False,
        smoke=False,
        resume=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Source-pinned, independently tuned optimizer quality campaign"
    )
    parser.add_argument(
        "--campaign",
        type=Path,
        default=Path("configs/experiments/optimizer_quality_campaign.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/optimizer-campaign/k3-tuning"),
    )
    parser.add_argument("--arm", action="append", default=[])
    parser.add_argument("--seed", action="append", type=int, default=[])
    parser.add_argument("--tokens", type=int, default=4_194_304)
    parser.add_argument(
        "--checkpoint-policy", choices=("none", "final_only", "full"), default="none"
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume-existing", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    source = assert_expected_checkout_source(root)
    if source.get("dirty"):
        raise RuntimeError("Optimizer quality campaigns require a clean source checkout")
    campaign_path = (root / args.campaign).resolve()
    output = (root / args.output).resolve()
    campaign = load_optimizer_campaign(campaign_path)
    arm_ids = tuple(args.arm or campaign["arms"].keys())
    unknown = sorted(set(arm_ids) - set(campaign["arms"]))
    if unknown:
        raise ValueError(f"Unknown optimizer arms: {unknown}")
    seeds = tuple(args.seed or [1337])
    if args.tokens <= 0:
        raise ValueError("--tokens must be positive")

    def resolve(value: str) -> Path:
        return (root / value).resolve()

    model_path = resolve(str(campaign["model"]))
    train_path = resolve(str(campaign["train"]))
    data_path = resolve(str(campaign["data"]))
    tokenizer = resolve(str(campaign["tokenizer"]))
    for required in (model_path, train_path, tokenizer):
        if not required.is_file():
            raise FileNotFoundError(required)
    _validate_data(data_path, root)
    AsterConfig.from_yaml(model_path)
    TrainConfig.from_yaml(train_path)
    cuda_toolchain = require_compatible_cuda_toolchain()

    output.mkdir(parents=True, exist_ok=True)
    pinned_model = output / "configs" / model_path.name
    pinned_model.parent.mkdir(parents=True, exist_ok=True)
    existing_manifest_path = output / "quality-campaign.json"
    existing_manifest = (
        json.loads(existing_manifest_path.read_text(encoding="utf-8"))
        if args.resume_existing and existing_manifest_path.is_file()
        else None
    )
    if args.resume_existing and existing_manifest is None:
        raise FileNotFoundError(
            f"--resume-existing requires {existing_manifest_path}"
        )
    if existing_manifest is None:
        pinned_model.write_bytes(model_path.read_bytes())
    elif not pinned_model.is_file():
        raise FileNotFoundError(f"Pinned optimizer model config is missing: {pinned_model}")
    execution_data = _absolute_data_config(
        data_path, root, output / "execution-data-absolute.yaml"
    )
    candidate_id = str(campaign["candidate_id"])
    matrix = [
        {"candidate_id": candidate_id, "execution_variant": arm_id}
        for arm_id in arm_ids
    ]
    selected_arms = {arm_id: campaign["arms"][arm_id] for arm_id in arm_ids}
    if existing_manifest is not None:
        _validate_resume_manifest(
            existing_manifest,
            arm_ids=arm_ids,
            arms=selected_arms,
            seeds=seeds,
            tokens=args.tokens,
            checkpoint_policy=args.checkpoint_policy,
        )
        if existing_manifest.get("model_sha256") != _sha256(pinned_model):
            raise ValueError("Pinned optimizer model config changed since campaign creation")
        resume_hashes = {
            "campaign_sha256": _sha256(campaign_path),
            "train_template_sha256": _sha256(train_path),
            "data_config_sha256": _sha256(data_path),
            "tokenizer_sha256": _sha256(tokenizer),
        }
        changed_inputs = {
            name: {"existing": existing_manifest.get(name), "current": current}
            for name, current in resume_hashes.items()
            if existing_manifest.get(name) != current
        }
        if changed_inputs:
            raise ValueError(f"Optimizer resume input hash mismatch: {changed_inputs}")
        manifest = copy.deepcopy(existing_manifest)
        manifest["status"] = "preflight_ok"
        manifest["resume_orchestrator_provenance"] = source
        manifest["cuda_toolchain_resume_check"] = cuda_toolchain
        manifest.setdefault("interrupted_attempts", [])
        manifest.setdefault("runs", {})
        execution_commit = str(manifest["source_provenance"]["git_commit"])
    else:
        manifest = {
            "schema_version": 2,
            "campaign_type": "optimizer_quality",
            "status": "preflight_ok",
            "source_provenance": source,
            "campaign": _portable(campaign_path, root),
            "campaign_sha256": _sha256(campaign_path),
            "model": _portable(pinned_model, root),
            "model_sha256": _sha256(pinned_model),
            "train_template": _portable(train_path, root),
            "train_template_sha256": _sha256(train_path),
            "data": _portable(data_path, root),
            "data_config_sha256": _sha256(data_path),
            "tokenizer": _portable(tokenizer, root),
            "tokenizer_sha256": _sha256(tokenizer),
            "candidate_id": candidate_id,
            "candidates": [candidate_id],
            "execution_variants": list(arm_ids),
            "execution_matrix": matrix,
            "seeds": list(seeds),
            "tokens_per_candidate": args.tokens,
            "smoke": False,
            "checkpoint_policy": args.checkpoint_policy,
            "cuda_toolchain": cuda_toolchain,
            "arms": selected_arms,
            "interrupted_attempts": [],
            "runs": {},
        }
        execution_commit = str(source["git_commit"])
    campaign_manifest = output / "quality-campaign.json"
    atomic_write_json(campaign_manifest, manifest)
    _refresh_analysis(campaign_manifest)
    if args.preflight_only:
        print(json.dumps(manifest, indent=2))
        return

    base_train = yaml.safe_load(train_path.read_text(encoding="utf-8")) or {}
    pinned = create_pinned_source_checkout(root, execution_commit)
    manifest["execution_checkout"] = pinned.manifest()
    atomic_write_json(campaign_manifest, manifest)
    try:
        for seed in seeds:
            for arm_id in arm_ids:
                run_dir = output / f"seed-{seed}" / candidate_id / arm_id
                existing = summarize_quality_run(run_dir)
                if existing["status"] == "ok" and existing["tokens_seen"] >= args.tokens:
                    key = f"{seed}:{candidate_id}:{arm_id}"
                    previous = manifest["runs"].get(key, {})
                    manifest["runs"][key] = {**previous, **existing}
                    atomic_write_json(campaign_manifest, manifest)
                    _refresh_analysis(campaign_manifest)
                    continue
                if (run_dir / "experiment.json").exists():
                    if args.checkpoint_policy != "none":
                        raise RuntimeError(
                            f"Incomplete checkpoint-retaining optimizer run: {run_dir}"
                        )
                    archived = archive_incomplete_quality_run(
                        run_dir,
                        output / "interrupted-attempts" / f"seed-{seed}" / candidate_id,
                    )
                    manifest["interrupted_attempts"].append(
                        {
                            "seed": seed,
                            "candidate_id": candidate_id,
                            "execution_variant": arm_id,
                            "archive": _portable(archived, root),
                            "summary": existing,
                        }
                    )
                train_payload = _arm_train_payload(
                    base_train,
                    run_dir=run_dir,
                    seed=seed,
                    max_tokens=args.tokens,
                    tokenizer=tokenizer,
                    common_overrides=campaign.get("common_train_overrides", {}),
                    arm=campaign["arms"][arm_id],
                    checkpoint_policy=args.checkpoint_policy,
                )
                generated = output / "train-configs" / f"seed-{seed}-{arm_id}.yaml"
                generated.parent.mkdir(parents=True, exist_ok=True)
                generated.write_text(
                    yaml.safe_dump(train_payload, sort_keys=False), encoding="utf-8"
                )
                command = [
                    sys.executable,
                    str(pinned.path / "scripts/train_pretrain.py"),
                    "--model",
                    str(pinned_model),
                    "--train",
                    str(generated),
                    "--data",
                    str(execution_data),
                ]
                environment = os.environ.copy()
                environment["PYTHONPATH"] = os.pathsep.join(
                    filter(None, (str(pinned.path / "src"), environment.get("PYTHONPATH")))
                )
                environment["ASTERLM_EXECUTION_VARIANT"] = arm_id
                completed = subprocess.run(
                    command, cwd=pinned.path, env=environment, check=False
                )
                summary = summarize_quality_run(run_dir)
                summary.update(
                    {
                        "candidate_id": candidate_id,
                        "execution_variant": arm_id,
                        "optimizer_family": campaign["arms"][arm_id]["family"],
                        "seed": seed,
                        "returncode": completed.returncode,
                        "train_config": _portable(generated, root),
                    }
                )
                manifest["runs"][f"{seed}:{candidate_id}:{arm_id}"] = summary
                manifest["status"] = (
                    "running" if completed.returncode == 0 else "partial_failure"
                )
                atomic_write_json(campaign_manifest, manifest)
                _refresh_analysis(campaign_manifest)
                if completed.returncode:
                    raise SystemExit(completed.returncode)
        manifest["status"] = "complete"
        atomic_write_json(campaign_manifest, manifest)
        _refresh_analysis(campaign_manifest)
    finally:
        pinned.close()


if __name__ == "__main__":
    main()
