from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def load_module():
    path = Path(__file__).parents[1] / "scripts" / "audit_posttrain_records.py"
    spec = importlib.util.spec_from_file_location("audit_posttrain_records", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_audit_counts_modes_lengths_and_duplicates(tmp_path: Path) -> None:
    module = load_module()
    records = [
        {
            "messages": [
                {"role": "user", "content": "Why?"},
                {"role": "assistant", "content": "<think>Because.</think> Done."},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Why?"},
                {"role": "assistant", "content": "<think>Because.</think> Done."},
            ]
        },
        {
            "chosen": [
                {"role": "user", "content": "Run it"},
                {"role": "assistant", "content": '{"commands": ["pytest"]}'},
            ],
            "rejected": [
                {"role": "user", "content": "Run it"},
                {"role": "assistant", "content": "I cannot assist"},
            ],
        },
    ]
    source = tmp_path / "records.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")

    report = module.audit_source(source, max_records=0, count_tokens=lambda text: len(text.split()))

    assert report["sampled_records"] == 3
    assert report["counts"]["exact_duplicate_records"] == 1
    assert report["counts"]["views_with_thinking_marker"] == 2
    assert report["counts"]["views_with_tool_protocol"] == 1
    assert report["counts"]["views_with_refusal_phrase"] == 1
    assert report["distributions"]["assistant_tokens"]["count"] == 4


def test_discover_manifest_root_selects_source_directories(tmp_path: Path) -> None:
    module = load_module()
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "b").mkdir()
    (tmp_path / "a").mkdir()

    assert module.discover_sources([str(tmp_path)]) == [tmp_path / "a", tmp_path / "b"]
