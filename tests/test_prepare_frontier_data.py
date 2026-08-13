from pathlib import Path

import pytest

from scripts.prepare_frontier_data import normalized_source_specs, parse_assignments


def test_separate_math_and_code_sources_preserve_fim_semantics() -> None:
    jobs = [
        ("fineweb_edu", Path("raw/fineweb"), "text"),
        ("nemotron_math", Path("raw/math"), "text"),
        ("nemotron_code", Path("raw/code"), "text"),
    ]
    specs = normalized_source_specs(
        jobs,
        weights={"fineweb_edu": 0.53, "nemotron_math": 0.05, "nemotron_code": 0.09},
        fim_sources={"nemotron_code"},
    )

    by_id = {str(spec["id"]): spec for spec in specs}
    assert by_id["nemotron_math"]["fim_rate"] == 0.0
    assert by_id["nemotron_code"]["fim_rate"] == 0.5
    assert sum(float(spec["weight"]) for spec in specs) == pytest.approx(1.0)


def test_assignment_parser_rejects_duplicates_and_missing_values() -> None:
    assert parse_assignments(["math=data/math"]) == {"math": "data/math"}
    with pytest.raises(ValueError, match="Duplicate"):
        parse_assignments(["math=a", "math=b"])
    with pytest.raises(ValueError, match="ID=VALUE"):
        parse_assignments(["math"])
