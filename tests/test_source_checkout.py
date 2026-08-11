from __future__ import annotations

import subprocess
from pathlib import Path

from asterlm.experiments.source_checkout import create_pinned_source_checkout

ROOT = Path(__file__).resolve().parents[1]


def test_pinned_checkout_is_clean_exact_and_removed(tmp_path: Path) -> None:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    pinned = create_pinned_source_checkout(
        ROOT, commit, temporary_parent=tmp_path
    )
    checkout = pinned.path
    manifest = pinned.manifest()
    assert manifest["observed_commit"] == commit
    assert manifest["dirty"] is False
    assert (checkout / "src" / "asterlm").is_dir()
    pinned.close()
    assert not checkout.exists()
