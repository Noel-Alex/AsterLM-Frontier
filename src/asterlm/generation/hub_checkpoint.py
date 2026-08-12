from __future__ import annotations

from pathlib import Path
from typing import Any


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
    return {run: sorted(values) for run, values in sorted(catalog.items())}


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
    """Download and verify one full Aster checkpoint from a private Hub repo."""

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
        except Exception:
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
