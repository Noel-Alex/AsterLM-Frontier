from __future__ import annotations

from pathlib import Path
from typing import Any


def checkpoint_order_key(name: str) -> tuple[int, int, str]:
    """Order checkpoint names by token milestone, then zero-padded step."""

    step = -1
    tokens = -1
    parts = name.split("-")
    if len(parts) >= 2 and parts[0] == "checkpoint":
        try:
            step = int(parts[1])
        except ValueError:
            pass
    for index, value in enumerate(parts[:-1]):
        if value == "tok":
            try:
                tokens = int(parts[index + 1])
            except ValueError:
                pass
    return tokens, step, name


def checkpoint_catalog(files: list[str]) -> dict[str, list[str]]:
    """Build a run/checkpoint catalog from a Hub repository file listing."""

    catalog: dict[str, set[str]] = {}
    for raw in files:
        parts = Path(raw).as_posix().split("/")
        if len(parts) < 5 or parts[0] != "runs" or parts[2] != "checkpoints":
            continue
        if parts[4] != "checkpoint_manifest.json":
            continue
        catalog.setdefault(parts[1], set()).add(parts[3])
    return {
        run: sorted(values, key=checkpoint_order_key)
        for run, values in sorted(catalog.items())
    }


def select_checkpoint_name(
    checkpoints: list[str],
    selector: str,
    *,
    latest: str | None = None,
) -> str:
    """Resolve exact, latest, final, or token-addressed checkpoint selectors."""

    if not checkpoints:
        raise ValueError("The selected Hub run has no complete checkpoints")
    value = selector.strip()
    if value == "latest":
        if latest and latest in checkpoints:
            return latest
        return checkpoints[-1]
    if value == "final":
        matches = [name for name in checkpoints if name.endswith("-final")]
        if not matches:
            raise ValueError("The selected Hub run has no final checkpoint")
        return matches[-1]
    if value.startswith("tokens:"):
        try:
            tokens = int(value.split(":", 1)[1].replace("_", "").replace(",", ""))
        except ValueError as exc:
            raise ValueError("Token selector must look like tokens:18400000000") from exc
        suffix = f"-tok-{tokens}"
        matches = [name for name in checkpoints if name.endswith(suffix)]
        if not matches:
            raise ValueError(f"No checkpoint exists at token milestone {tokens:,}")
        return matches[-1]
    if value in checkpoints:
        return value
    raise ValueError(
        f"Unknown checkpoint selector {selector!r}; use latest, final, tokens:N, or an exact name"
    )


def select_newer_checkpoint(
    local: tuple[Path, dict[str, Any]] | None,
    remote: tuple[str, dict[str, Any]] | None,
) -> str:
    """Select local, remote, or none using durable token/step counters."""

    def rank(manifest: dict[str, Any]) -> tuple[int, int]:
        return int(manifest.get("tokens_seen", -1)), int(manifest.get("step", -1))

    if local is None and remote is None:
        return "none"
    if local is None:
        return "remote"
    if remote is None:
        return "local"
    return "remote" if rank(remote[1]) > rank(local[1]) else "local"


def checkpoint_model_compatible(
    requested: Any,
    saved: Any,
) -> tuple[bool, list[str]]:
    """Compare checkpoint-defining model fields while allowing execution tuning."""

    requested_values = requested.to_dict()
    saved_values = saved.to_dict()
    # These fields alter storage/recomputation or the maximum admitted input, not
    # learned tensor shapes/equations. They may be tuned per physical GPU/stage.
    ignored = {
        "max_seq_len",
        "gradient_checkpointing",
        "checkpoint_segment_size",
        "lm_loss_chunk_size",
        "lm_loss_backend",
        "linear_ce_chunking_method",
        "linear_ce_acc_policy",
    }
    if requested_values.get("kda_backend") == "auto":
        requested_values["kda_backend"] = saved_values.get("kda_backend")
    mismatches = sorted(
        key
        for key in requested_values.keys() | saved_values.keys()
        if key not in ignored and requested_values.get(key) != saved_values.get(key)
    )
    return not mismatches, mismatches


def latest_local_checkpoint(
    output_dir: str | Path,
    explicit: str | Path | None = None,
) -> tuple[Path, dict[str, Any]] | None:
    from asterlm.training.checkpoint import resolve_checkpoint, verify_checkpoint

    root = Path(explicit) if explicit is not None else Path(output_dir)
    if not root.exists():
        return None
    candidate = resolve_checkpoint(root)
    if not candidate.is_dir() or not (candidate / "checkpoint_manifest.json").is_file():
        return None
    return candidate, verify_checkpoint(candidate)


def latest_hub_checkpoint_manifest(
    repo_id: str,
    *,
    run: str,
    revision: str = "main",
    cache_dir: str | Path | None = None,
    token: str | bool | None = None,
    requested_model: Any | None = None,
) -> tuple[str, dict[str, Any]] | None:
    """Download only small manifests and identify the newest complete remote state."""

    import json

    from huggingface_hub import hf_hub_download

    catalog = list_hub_checkpoints(repo_id, revision=revision, token=token)
    records: list[tuple[str, dict[str, Any]]] = []
    for name in catalog.get(run, []):
        path = hf_hub_download(
            repo_id=repo_id,
            filename=f"runs/{run}/checkpoints/{name}/checkpoint_manifest.json",
            repo_type="model",
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir else None,
            token=token,
        )
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            continue
        if requested_model is not None:
            from asterlm.config import AsterConfig

            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=f"runs/{run}/checkpoints/{name}/model_config.yaml",
                repo_type="model",
                revision=revision,
                cache_dir=str(cache_dir) if cache_dir else None,
                token=token,
            )
            saved_model = AsterConfig.from_yaml(model_path)
            compatible, mismatches = checkpoint_model_compatible(
                requested_model, saved_model
            )
            if not compatible:
                continue
            manifest = {**manifest, "model_compatibility_mismatches": mismatches}
        records.append((name, manifest))
    if not records:
        return None
    return max(
        records,
        key=lambda item: (
            int(item[1].get("tokens_seen", -1)),
            int(item[1].get("step", -1)),
            checkpoint_order_key(item[0]),
        ),
    )


def resolve_hub_auto_resume(
    *,
    output_dir: str | Path,
    repo_id: str,
    revision: str = "main",
    cache_dir: str | Path | None = None,
    token: str | bool | None = None,
    local_checkpoint: str | Path | None = None,
    requested_model: Any | None = None,
) -> tuple[Path | None, dict[str, Any]]:
    """Choose the newest verified local/Hub state and download only when needed."""

    output = Path(output_dir)
    run = output.name
    local = latest_local_checkpoint(output, explicit=local_checkpoint)
    remote = latest_hub_checkpoint_manifest(
        repo_id,
        run=run,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        requested_model=requested_model,
    )
    choice = select_newer_checkpoint(local, remote)
    if choice == "none":
        return None, {"choice": "none", "run": run, "repo_id": repo_id}
    if choice == "local":
        assert local is not None
        return local[0], {
            "choice": "local",
            "run": run,
            "repo_id": repo_id,
            "checkpoint": local[0].name,
            "tokens_seen": local[1].get("tokens_seen"),
            "step": local[1].get("step"),
        }
    assert remote is not None
    checkpoint, record = download_hub_checkpoint(
        repo_id,
        run=run,
        selector=remote[0],
        revision=revision,
        cache_dir=cache_dir,
        token=token,
    )
    return checkpoint, {"choice": "remote", **record}


def list_hub_checkpoints(
    repo_id: str,
    *,
    revision: str = "main",
    token: str | bool | None = None,
) -> dict[str, list[str]]:
    from huggingface_hub import HfApi

    files = HfApi(token=token).list_repo_files(
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
    )
    return checkpoint_catalog(list(files))


def download_hub_checkpoint(
    repo_id: str,
    *,
    run: str | None = None,
    selector: str = "latest",
    revision: str = "main",
    cache_dir: str | Path | None = None,
    token: str | bool | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Download and verify one full Aster checkpoint from a Hub repo."""

    from huggingface_hub import hf_hub_download, snapshot_download

    catalog = list_hub_checkpoints(repo_id, revision=revision, token=token)
    if run is None:
        if len(catalog) != 1:
            choices = ", ".join(catalog) or "none"
            raise ValueError(f"--hub-run is required; available runs: {choices}")
        run = next(iter(catalog))
    if run not in catalog:
        raise ValueError(f"Unknown Hub run {run!r}; available runs: {', '.join(catalog)}")

    latest: str | None = None
    if selector == "latest":
        try:
            pointer = hf_hub_download(
                repo_id=repo_id,
                filename=f"runs/{run}/latest.txt",
                repo_type="model",
                revision=revision,
                cache_dir=str(cache_dir) if cache_dir else None,
                token=token,
            )
            latest = Path(pointer).read_text(encoding="utf-8").strip()
        except Exception:  # noqa: BLE001 - legacy repositories may lack the pointer
            # A legacy upload may not have the portable pointer; padded step names
            # still provide a deterministic fallback.
            latest = None
    checkpoint_name = select_checkpoint_name(catalog[run], selector, latest=latest)
    prefix = f"runs/{run}/checkpoints/{checkpoint_name}"
    snapshot_root = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
            allow_patterns=[f"{prefix}/**"],
            cache_dir=str(cache_dir) if cache_dir else None,
            token=token,
        )
    )
    checkpoint = snapshot_root / prefix
    from asterlm.training.checkpoint import verify_checkpoint

    manifest = verify_checkpoint(checkpoint)
    return checkpoint, {
        "repo_id": repo_id,
        "revision": revision,
        "run": run,
        "selector": selector,
        "checkpoint": checkpoint_name,
        "tokens_seen": manifest.get("tokens_seen"),
        "step": manifest.get("step"),
        "local_path": str(checkpoint),
    }
