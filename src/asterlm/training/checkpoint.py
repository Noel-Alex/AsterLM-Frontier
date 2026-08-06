from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

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
) -> Path:
    root = Path(output_dir)
    safe_tag = "" if tag is None else "-" + "".join(
        char if char.isalnum() or char in {"-", "_"} else "-" for char in tag
    ).strip("-")
    checkpoint_dir = root / f"checkpoint-{step:08d}{safe_tag}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_path = checkpoint_dir / "model.safetensors"
    try:
        from safetensors.torch import save_model

        save_model(model, str(model_path), metadata={"format": "pt", "architecture": "AsterLM"})
    except Exception:
        model_path = checkpoint_dir / "model.pt"
        torch.save(model.state_dict(), model_path)

    torch.save(
        {
            "step": step,
            "tokens_seen": tokens_seen,
            "optimizer": optimizer.state_dict(),
            "rng": _rng_state(),
        },
        checkpoint_dir / "trainer_state.pt",
    )
    saved_model_config = model_config.to_dict()
    # `auto` is convenient at experiment creation but unsafe inside a checkpoint:
    # installing/removing FLA later would otherwise instantiate a different parameterization.
    saved_model_config["kda_backend"] = "fla" if bool(getattr(model, "uses_fla", False)) else "torch"
    (checkpoint_dir / "model_config.yaml").write_text(
        yaml.safe_dump({"model": saved_model_config}, sort_keys=False), encoding="utf-8"
    )
    (checkpoint_dir / "train_config.yaml").write_text(
        yaml.safe_dump({"train": train_config.to_dict()}, sort_keys=False), encoding="utf-8"
    )
    (checkpoint_dir / "checkpoint_manifest.json").write_text(
        json.dumps(
            {
                "step": step,
                "tokens_seen": tokens_seen,
                "reason": reason,
                "permanent": permanent,
                "model_file": model_path.name,
                "model_bytes": model_path.stat().st_size,
                "trainer_state_bytes": (checkpoint_dir / "trainer_state.pt").stat().st_size,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if permanent:
        (checkpoint_dir / "KEEP").write_text(reason + "\n", encoding="utf-8")
    (root / "latest.txt").write_text(str(checkpoint_dir.resolve()), encoding="utf-8")

    # Permanent token milestones and final checkpoints are never removed by rolling
    # retention. Only ordinary periodic checkpoints count toward keep_last.
    rolling = [
        checkpoint
        for checkpoint in sorted(root.glob("checkpoint-*"))
        if not (checkpoint / "KEEP").exists()
    ]
    for old in rolling[:-keep_last] if keep_last > 0 else []:
        shutil.rmtree(old, ignore_errors=True)
    return checkpoint_dir


def resolve_checkpoint(path: str | Path) -> Path:
    path = Path(path)
    if path.is_dir() and (path / "latest.txt").exists():
        target = Path((path / "latest.txt").read_text(encoding="utf-8").strip())
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
    safe = checkpoint / "model.safetensors" if checkpoint.is_dir() else checkpoint
    if safe.suffix == ".safetensors" and safe.exists():
        from safetensors.torch import load_model

        missing, unexpected = load_model(model, str(safe), strict=strict)
        if strict and (missing or unexpected):
            raise RuntimeError(f"Checkpoint mismatch; missing={missing}, unexpected={unexpected}")
    else:
        pt = checkpoint / "model.pt" if checkpoint.is_dir() else checkpoint
        state = torch.load(pt, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=strict)
    return checkpoint


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
