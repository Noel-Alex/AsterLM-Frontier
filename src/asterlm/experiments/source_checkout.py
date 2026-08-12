from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )


@dataclass(slots=True)
class PinnedSourceCheckout:
    repository: Path
    commit: str
    temporary_root: Path
    path: Path
    _closed: bool = False

    def manifest(self) -> dict[str, Any]:
        observed = _git(self.path, "rev-parse", "HEAD").stdout.strip()
        status = _git(self.path, "status", "--porcelain=v1").stdout.splitlines()
        return {
            "repository": str(self.repository),
            "commit": self.commit,
            "path": str(self.path),
            "observed_commit": observed,
            "dirty": bool(status),
            "status_porcelain": status,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            _git(self.repository, "worktree", "remove", "--force", str(self.path))
        except (OSError, subprocess.SubprocessError):
            pass
        shutil.rmtree(self.temporary_root, ignore_errors=True)

    def __enter__(self) -> PinnedSourceCheckout:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def create_pinned_source_checkout(
    repository: str | Path,
    commit: str,
    *,
    temporary_parent: str | Path | None = None,
) -> PinnedSourceCheckout:
    repo = Path(repository).resolve()
    parent = Path(temporary_parent).resolve() if temporary_parent else None
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f"aster-source-{commit[:12]}-", dir=parent)
    )
    checkout = temporary_root / "checkout"
    try:
        _git(repo, "worktree", "add", "--detach", str(checkout), commit)
        pinned = PinnedSourceCheckout(repo, commit, temporary_root, checkout)
        manifest = pinned.manifest()
        if manifest["observed_commit"] != commit or manifest["dirty"]:
            raise RuntimeError(f"Pinned checkout verification failed: {manifest}")
        return pinned
    except BaseException:
        try:
            _git(repo, "worktree", "remove", "--force", str(checkout))
        except (OSError, subprocess.SubprocessError):
            pass
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
