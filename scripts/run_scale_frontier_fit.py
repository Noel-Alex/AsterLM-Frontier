#!/usr/bin/env python3
"""Run source-pinned full-training-state fit probes across the K3 scale frontier."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from asterlm.artifacts import atomic_write_json
from asterlm.cuda_allocator import cuda_allocator_environment
from asterlm.experiments.source_checkout import create_pinned_source_checkout
from asterlm.source_provenance import assert_expected_checkout_source

DEFAULT_MODELS = (
    "configs/model/aster_k3_latentmoe_868m_a483m.yaml",
    "configs/model/aster_k3_latentmoe_1p45b_a568m.yaml",
    "configs/model/aster_k3_latentmoe_1p95b_a766m.yaml",
)


@dataclass(frozen=True, slots=True)
class FitTrial:
    model: str
    sequence: int
    micro_batch: int
    accumulation: int
    optimizer: str
    moe_implementation: str

    @property
    def trial_id(self) -> str:
        model = Path(self.model).stem
        return (
            f"{model}-s{self.sequence}-b{self.micro_batch}-a{self.accumulation}-"
            f"{self.optimizer}-{self.moe_implementation}"
        )


def build_trial_plan(
    models: list[str],
    *,
    sequences: list[int],
    batches: list[int],
    optimizers: list[str],
    target_update_tokens: int,
    moe_implementation: str,
) -> list[FitTrial]:
    trials: list[FitTrial] = []
    for model in models:
        for sequence in sequences:
            for optimizer in optimizers:
                for batch in sorted(set(batches), reverse=True):
                    accumulation = max(1, math.ceil(target_update_tokens / (sequence * batch)))
                    trials.append(
                        FitTrial(
                            model=model,
                            sequence=sequence,
                            micro_batch=batch,
                            accumulation=accumulation,
                            optimizer=optimizer,
                            moe_implementation=moe_implementation,
                        )
                    )
    return trials


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--sequence", action="append", type=int, default=[])
    parser.add_argument("--batch", action="append", type=int, default=[])
    parser.add_argument("--optimizer", action="append", default=[])
    parser.add_argument(
        "--moe-implementation",
        choices=("reference", "cutlass", "torch_grouped"),
        default="cutlass",
    )
    parser.add_argument("--target-update-tokens", type=int, default=16_384)
    parser.add_argument("--no-muon-per-head", action="store_true")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--trial-timeout", type=float, default=1800.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/architecture-campaign/k3-scale-frontier-fit"),
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    source = assert_expected_checkout_source(root)
    if source.get("dirty"):
        raise RuntimeError("Scale-frontier fit probes require a clean source-pinned checkout")
    output = args.output if args.output.is_absolute() else root / args.output
    output.mkdir(parents=True, exist_ok=True)

    models = list(args.model or DEFAULT_MODELS)
    sequences = list(args.sequence or [2048])
    batches = list(args.batch or [4, 2, 1])
    optimizers = list(args.optimizer or ["adamw", "torchao_adamw8bit"])
    if any(value <= 0 for value in [*sequences, *batches, args.target_update_tokens]):
        raise ValueError("sequence, batch, and target-update-tokens must be positive")
    model_records: dict[str, dict[str, Any]] = {}
    for relative in models:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        model_records[relative] = {"sha256": _sha256(path)}

    trials = build_trial_plan(
        models,
        sequences=sequences,
        batches=batches,
        optimizers=optimizers,
        target_update_tokens=args.target_update_tokens,
        moe_implementation=args.moe_implementation,
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "prepared",
        "created_utc": datetime.now(UTC).isoformat(),
        "source_provenance": source,
        "models": model_records,
        "protocol": {
            "sequences": sequences,
            "batches_descending": sorted(set(batches), reverse=True),
            "optimizers": optimizers,
            "target_update_tokens": args.target_update_tokens,
            "steps": args.steps,
            "warmup": args.warmup,
            "moe_implementation": args.moe_implementation,
            "muon_per_head": not args.no_muon_per_head,
            "trial_timeout_seconds": args.trial_timeout,
        },
        "planned_trials": [asdict(trial) | {"trial_id": trial.trial_id} for trial in trials],
        "trials": [],
    }
    manifest_path = output / "scale-fit-manifest.json"
    atomic_write_json(manifest_path, manifest)

    pinned = create_pinned_source_checkout(root, str(source["git_commit"]))
    manifest["execution_checkout"] = pinned.manifest()
    manifest["status"] = "running"
    atomic_write_json(manifest_path, manifest)
    try:
        for trial in trials:
            result_path = output / f"{trial.trial_id}.json"
            command = [
                sys.executable,
                str(pinned.path / "scripts" / "profile_training.py"),
                "--model",
                str(pinned.path / trial.model),
                "--train-config",
                str(pinned.path / "configs" / "train" / "probe_memory_matrix.yaml"),
                "--sequence",
                str(trial.sequence),
                "--batch",
                str(trial.micro_batch),
                "--accum",
                str(trial.accumulation),
                "--steps",
                str(args.steps),
                "--warmup",
                str(args.warmup),
                "--optimizer",
                trial.optimizer,
                "--precision",
                "amp",
                "--moe-implementation",
                trial.moe_implementation,
            ]
            if trial.optimizer == "muon_adamw" and not args.no_muon_per_head:
                command.append("--muon-per-head")
            command += [
                "--json",
                str(result_path),
            ]
            environment = cuda_allocator_environment(os.environ)
            environment["PYTHONPATH"] = str(pinned.path / "src")
            started = datetime.now(UTC).isoformat()
            try:
                completed = subprocess.run(
                    command,
                    cwd=pinned.path,
                    env=environment,
                    check=False,
                    timeout=args.trial_timeout,
                )
                returncode = completed.returncode
                failure = None
            except subprocess.TimeoutExpired:
                returncode = 124
                failure = "timeout"
            payload = (
                json.loads(result_path.read_text(encoding="utf-8"))
                if result_path.is_file()
                else {}
            )
            manifest["trials"].append(
                {
                    **asdict(trial),
                    "trial_id": trial.trial_id,
                    "started_utc": started,
                    "returncode": returncode,
                    "status": payload.get("status", failure or "missing_result"),
                    "result": result_path.relative_to(root).as_posix(),
                    "summary": payload.get("summary"),
                    "error": payload.get("error", failure),
                }
            )
            atomic_write_json(manifest_path, manifest)
    finally:
        pinned.close()
    manifest["status"] = "complete"
    atomic_write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
