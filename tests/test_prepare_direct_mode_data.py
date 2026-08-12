from __future__ import annotations

import json

from scripts.prepare_direct_mode_data import converted


def _write_records(path, source: str, count: int) -> None:
    rows = [
        {
            "source": source,
            "messages": [
                {"role": "user", "content": f"question {source} {index}"},
                {"role": "assistant", "content": f"answer {source} {index}"},
            ],
        }
        for index in range(count)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_direct_mode_conversion_balances_inputs_before_global_limit(tmp_path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    _write_records(first, "first", 4)
    _write_records(second, "second", 4)
    stats = {}

    rows = list(converted([str(first), str(second)], max_records=4, stats=stats))

    assert [row["source"] for row in rows] == ["first", "second", "first", "second"]
    assert all(row["messages"][-1]["content"].startswith("<|direct|>\n<answer>") for row in rows)
    assert stats["sources"][str(first)]["emitted"] == 2
    assert stats["sources"][str(second)]["emitted"] == 2


def test_direct_mode_conversion_continues_after_short_source_exhausts(tmp_path) -> None:
    short = tmp_path / "short.jsonl"
    long = tmp_path / "long.jsonl"
    _write_records(short, "short", 1)
    _write_records(long, "long", 5)

    rows = list(converted([str(short), str(long)], max_records=5))

    assert len(rows) == 5
    assert sum(row["source"] == "short" for row in rows) == 1
    assert sum(row["source"] == "long" for row in rows) == 4
