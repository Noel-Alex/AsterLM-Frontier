from __future__ import annotations

import json
import os
import random
import shutil
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from asterlm.artifacts import (
    artifact_record,
    atomic_write_json,
    atomic_write_text,
    fsync_directory,
    fsync_file,
    sha256_file,
)
from asterlm.config import AsterConfig, TrainConfig
from asterlm.optim.hybrid import HybridOptimizer, SingleOptimizerAdapter


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


@contextmanager
def checkpoint_compatible_parameter_storage(
    model: torch.nn.Module,
) -> Iterator[dict[str, float]]:
    """Materialize partial shared views for Safetensors, then restore packing."""

    materialize = getattr(model, "materialize_grouped_expert_storage", None)
    repack = getattr(model, "pack_grouped_expert_storage", None)
    materialized = materialize() if callable(materialize) else {}
    try:
        yield materialized
    finally:
        if materialized and callable(repack):
            repack()


def save_checkpoint(
    output_dir: str | Path,
    step: int,
    model: torch.nn.Module,
    optimizer: HybridOptimizer | SingleOptimizerAdapter,
    model_config: AsterConfig,
    train_config: TrainConfig,
    tokens_seen: int,
    keep_last: int = 3,
    *,
    tag: str | None = None,
    permanent: bool = False,
    reason: str = "periodic",
    data_state: dict[str, Any] | None = None,
    prune: bool = True,
) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    safe_tag = "" if tag is None else "-" + "".join(
        char if char.isalnum() or char in {"-", "_"} else "-" for char in tag
    ).strip("-")
    checkpoint_dir = root / f"checkpoint-{step:08d}{safe_tag}"
    if checkpoint_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing checkpoint {checkpoint_dir}; use a unique tag"
        )
    staging = root / f".{checkpoint_dir.name}.partial-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    atomic_write_json(
        staging / "partial_manifest.json",
        {
            "schema_version": 2,
            "status": "writing",
            "step": step,
            "tokens_seen": tokens_seen,
            "reason": reason,
        },
    )
    try:
        model_path = staging / "model.safetensors"
        with checkpoint_compatible_parameter_storage(model):
            try:
                from safetensors.torch import save_model
            except ImportError:
                model_path = staging / "model.pt"
                torch.save(model.state_dict(), model_path)
            else:
                save_model(
                    model,
                    str(model_path),
                    metadata={"format": "pt", "architecture": "AsterLM"},
                )

        trainer_state = staging / "trainer_state.pt"
        torch.save(
            {
                "step": step,
                "tokens_seen": tokens_seen,
                "optimizer": optimizer.state_dict(),
                "rng": _rng_state(),
            },
            trainer_state,
        )
        saved_model_config = model_config.to_dict()
        # `auto` is convenient at experiment creation but unsafe inside a checkpoint:
        # installing/removing FLA later would otherwise instantiate a different parameterization.
        saved_model_config["kda_backend"] = (
            "fla" if bool(getattr(model, "uses_fla", False)) else "torch"
        )
        model_config_path = staging / "model_config.yaml"
        train_config_path = staging / "train_config.yaml"
        atomic_write_text(
            model_config_path,
            yaml.safe_dump({"model": saved_model_config}, sort_keys=False),
        )
        atomic_write_text(
            train_config_path,
            yaml.safe_dump({"train": train_config.to_dict()}, sort_keys=False),
        )

        durable_files = [model_path, trainer_state, model_config_path, train_config_path]
        data_state_path: Path | None = None
        if data_state is not None:
            data_state_path = staging / "data_state.pt"
            torch.save(data_state, data_state_path)
            durable_files.append(data_state_path)
        for path in durable_files:
            fsync_file(path)
        artifacts = [artifact_record(path, relative_to=staging) for path in durable_files]
        manifest = {
            "schema_version": 2,
            "status": "complete",
            "step": step,
            "tokens_seen": tokens_seen,
            "reason": reason,
            "permanent": permanent,
            "model_file": model_path.name,
            "model_bytes": model_path.stat().st_size,
            "trainer_state_bytes": trainer_state.stat().st_size,
            "data_state_file": data_state_path.name if data_state_path else None,
            "data_state_bytes": data_state_path.stat().st_size if data_state_path else None,
            "artifacts": artifacts,
            "resume_state": {
                "model": True,
                "optimizer": True,
                "scheduler": True,
                "rng_python": True,
                "rng_numpy": True,
                "rng_torch_cpu": True,
                "rng_torch_cuda": torch.cuda.is_available(),
                "global_step": True,
                "tokens_seen": True,
                "data_pipeline": data_state_path is not None,
            },
        }
        atomic_write_json(staging / "checkpoint_manifest.json", manifest)
        (staging / "partial_manifest.json").unlink(missing_ok=True)
        if permanent:
            atomic_write_text(staging / "KEEP", reason + "\n")
        fsync_directory(staging)
        os.replace(staging, checkpoint_dir)
        fsync_directory(root)
        # Keep the pointer portable across Windows, WSL, Modal, GCP, and Hub
        # round-trips.  `resolve_checkpoint` interprets relative pointers against
        # the run directory; absolute pointers from older checkpoints remain
        # supported for backwards compatibility.
        atomic_write_text(root / "latest.txt", checkpoint_dir.name + "\n")
    except BaseException:
        # Leave the hidden partial directory for power-loss/failure forensics. It is
        # never considered loadable because it has no complete published manifest.
        raise

    if prune:
        prune_rolling_checkpoints(root, keep_last=keep_last)
    return checkpoint_dir


def prune_rolling_checkpoints(
    output_dir: str | Path,
    *,
    keep_last: int,
    pyramid_levels: int = 0,
    protected: set[Path] | None = None,
) -> list[Path]:
    """Prune transient checkpoints only after their durability policy is satisfied.

    Checkpoint creation and retention are deliberately separate operations.  A
    caller that promises remote durability can save with ``prune=False``, upload
    and hash-verify the checkpoint, and only then invoke this function.
    """

    if keep_last < 0 or pyramid_levels < 0:
        raise ValueError("checkpoint retention values must be non-negative")
    root = Path(output_dir)
    protected_resolved = {path.resolve() for path in (protected or set())}
    # Permanent token milestones and final checkpoints are never removed by rolling
    # retention. Only ordinary periodic checkpoints count toward keep_last.
    rolling = [
        checkpoint
        for checkpoint in sorted(root.glob("checkpoint-*"))
        if not (checkpoint / "KEEP").exists()
    ]
    recent_start = max(0, len(rolling) - keep_last) if keep_last > 0 else len(rolling)
    keep = {path.resolve() for path in rolling[recent_start:]}
    cursor = recent_start
    width = max(1, keep_last)
    for _ in range(pyramid_levels):
        if cursor <= 0:
            break
        start = max(0, cursor - width)
        # The newest checkpoint in each exponentially widening age band gives a
        # monotonic history: dense near the run head, increasingly sparse behind it.
        keep.add(rolling[cursor - 1].resolve())
        cursor = start
        width *= 2
    candidates = [path for path in rolling if path.resolve() not in keep]
    removed: list[Path] = []
    for old in candidates:
        if old.resolve() in protected_resolved:
            continue
        shutil.rmtree(old)
        removed.append(old)
    return removed


def verify_checkpoint(checkpoint: str | Path) -> dict[str, Any]:
    """Validate the completion marker, sizes, and hashes before loading or upload."""
    root = Path(checkpoint)
    manifest_path = root / "checkpoint_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"Checkpoint is missing its manifest: {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError(f"Checkpoint is not complete: {root}")
    for artifact in manifest.get("artifacts", []):
        path = root / str(artifact["path"])
        if not path.is_file():
            raise RuntimeError(f"Checkpoint artifact is missing: {path}")
        if path.stat().st_size != int(artifact["size_bytes"]):
            raise RuntimeError(f"Checkpoint artifact size mismatch: {path}")
        actual = sha256_file(path)
        if actual != artifact["sha256"]:
            raise RuntimeError(f"Checkpoint artifact hash mismatch: {path}")
    return manifest


def resolve_checkpoint(path: str | Path) -> Path:
    path = Path(path)
    if path.is_dir() and (path / "latest.txt").exists():
        target = Path((path / "latest.txt").read_text(encoding="utf-8").strip())
        if not target.is_absolute():
            target = path / target
        if target.exists():
            return target
    return path


def pin_kda_backend_from_checkpoint(model_config: AsterConfig, checkpoint: str | Path) -> Path:
    """Pin an `auto` config to the backend recorded by a checkpoint.

    The FLA and reference KDA implementations have different internal parameterizations,
    so silently resolving `auto` differently is never a valid checkpoint conversion.
    """
    resolved = resolve_checkpoint(checkpoint)
    config_path = resolved / "model_config.yaml" if resolved.is_dir() else None
    if config_path is None or not config_path.exists():
        return resolved
    saved = AsterConfig.from_yaml(config_path)
    if saved.kda_backend not in {"fla", "torch"}:
        return resolved
    if model_config.kda_backend == "auto":
        model_config.kda_backend = saved.kda_backend
    elif model_config.kda_backend != saved.kda_backend:
        raise ValueError(
            f"Checkpoint requires kda_backend={saved.kda_backend!r}, but the supplied "
            f"model config requests {model_config.kda_backend!r}"
        )
    return resolved


def load_model_weights(model: torch.nn.Module, checkpoint: str | Path, strict: bool = True) -> Path:
    checkpoint = resolve_checkpoint(checkpoint)
    if checkpoint.is_dir() and (checkpoint / "checkpoint_manifest.json").exists():
        verify_checkpoint(checkpoint)
    safe = checkpoint / "model.safetensors" if checkpoint.is_dir() else checkpoint
    if safe.suffix == ".safetensors" and safe.exists():
        from safetensors.torch import load_model

        with checkpoint_compatible_parameter_storage(model):
            missing, unexpected = load_model(model, str(safe), strict=strict)
        if strict and (missing or unexpected):
            raise RuntimeError(f"Checkpoint mismatch; missing={missing}, unexpected={unexpected}")
    else:
        pt = checkpoint / "model.pt" if checkpoint.is_dir() else checkpoint
        state = torch.load(pt, map_location="cpu", weights_only=True)
        with checkpoint_compatible_parameter_storage(model):
            model.load_state_dict(state, strict=strict)
    return checkpoint


def load_data_state(checkpoint: str | Path) -> dict[str, Any] | None:
    """Load the independently hashed data cursor from a complete checkpoint."""
    resolved = resolve_checkpoint(checkpoint)
    if not resolved.is_dir():
        return None
    manifest = verify_checkpoint(resolved)
    filename = manifest.get("data_state_file")
    if not filename:
        return None
    state = torch.load(resolved / str(filename), map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise RuntimeError("Checkpoint data_state is not a mapping")
    return state


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: HybridOptimizer | SingleOptimizerAdapter | None,
    checkpoint: str | Path,
    restore_rng: bool = True,
) -> tuple[int, int]:
    checkpoint = load_model_weights(model, checkpoint)
    state_path = checkpoint / "trainer_state.pt"
    if not state_path.exists():
        return 0, 0
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if restore_rng and state.get("rng") is not None:
        _restore_rng(state["rng"])
    return int(state.get("step", 0)), int(state.get("tokens_seen", 0))
