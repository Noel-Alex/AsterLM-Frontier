from __future__ import annotations

from pathlib import Path
from typing import Any

from asterlm.experiments.registry import git_provenance


def imported_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def source_provenance(expected_repo_root: str | Path) -> dict[str, Any]:
    expected = Path(expected_repo_root).resolve()
    imported = imported_repo_root()
    return {
        "expected_repo_root": str(expected),
        "imported_repo_root": str(imported),
        "source_matches_expected_checkout": imported == expected,
        **git_provenance(imported),
    }


def assert_expected_checkout_source(expected_repo_root: str | Path) -> dict[str, Any]:
    provenance = source_provenance(expected_repo_root)
    if not provenance["source_matches_expected_checkout"]:
        raise RuntimeError(
            "Imported AsterLM source does not match the executing checkout: "
            f"imported={provenance['imported_repo_root']}, "
            f"expected={provenance['expected_repo_root']}. Reinstall the checkout "
            "editable or put its src directory first on PYTHONPATH before benchmarking."
        )
    return provenance


def assert_current_checkout_source() -> dict[str, Any] | None:
    """Validate source when invoked from an Aster checkout; allow library use elsewhere."""

    cwd = Path.cwd().resolve()
    if (cwd / "pyproject.toml").is_file() and (cwd / "src" / "asterlm").is_dir():
        return assert_expected_checkout_source(cwd)
    return None
