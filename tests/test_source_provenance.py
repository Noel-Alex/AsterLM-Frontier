from __future__ import annotations

from pathlib import Path

import pytest

from asterlm.source_provenance import assert_expected_checkout_source, imported_repo_root


def test_imported_source_matches_test_checkout() -> None:
    root = Path(__file__).resolve().parents[1]
    result = assert_expected_checkout_source(root)
    assert result["source_matches_expected_checkout"]
    assert Path(result["imported_repo_root"]) == root


def test_mismatched_checkout_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="does not match"):
        assert_expected_checkout_source(tmp_path)


def test_imported_root_contains_project_metadata() -> None:
    assert (imported_repo_root() / "pyproject.toml").is_file()
