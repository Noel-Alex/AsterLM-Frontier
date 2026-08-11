#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from asterlm.data.tokenizer import normalize_messages
from asterlm.reasoning.io import atomic_write_jsonl, iter_json_records


def stable_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def messages_from_record(record: dict[str, Any]) -> list[dict[str, str]]:
    for key in ("messages", "conversations", "chosen"):
        value = record.get(key)
        if isinstance(value, list):
            messages = normalize_messages(value)
            if any(message["role"] == "assistant" and message["content"].strip() for message in messages):
                return messages
    return []


def direct_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for message in messages:
        item = dict(message)
        if item["role"] == "assistant":
            content = item["content"].strip()
            if not content:
                continue
            if not content.startswith("<|direct|>"):
                content = f"<|direct|>\n<answer>{content}</answer>"
            item["content"] = content
        output.append(item)
    return output


def converted(
    inputs: list[str],
    max_records: int,
    stats: dict[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    emitted = 0
    seen: set[str] = set()
    source_stats = {
        input_path: {"read": 0, "emitted": 0, "duplicates": 0, "invalid": 0}
        for input_path in inputs
    }
    active = [(input_path, iter(iter_json_records(input_path))) for input_path in inputs]
    while active and (not max_records or emitted < max_records):
        next_active = []
        for input_path, records in active:
            if max_records and emitted >= max_records:
                break
            try:
                record = next(records)
            except StopIteration:
                continue
            next_active.append((input_path, records))
            source_stats[input_path]["read"] += 1
            messages = messages_from_record(record)
            if not messages:
                source_stats[input_path]["invalid"] += 1
                continue
            rendered = json.dumps(messages, sort_keys=True, ensure_ascii=False)
            key = stable_id(rendered)
            if key in seen:
                source_stats[input_path]["duplicates"] += 1
                continue
            seen.add(key)
            yield {
                "id": key,
                "messages": direct_messages(messages),
                "source": str(record.get("source", record.get("_dataset", "unknown"))),
                "mode": "direct",
            }
            emitted += 1
            source_stats[input_path]["emitted"] += 1
        active = next_active
    if stats is not None:
        stats.update({"records": emitted, "sources": source_stats})


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert broad instruction records to Aster direct-mode SFT")
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", default="data/reasoning/direct_mode_sft.jsonl")
    parser.add_argument("--max-records", type=int, default=150000)
    parser.add_argument("--stats", default="data/reasoning/direct_mode_stats.json")
    args = parser.parse_args()
    stats: dict[str, Any] = {}
    count = atomic_write_jsonl(args.output, converted(args.inputs, args.max_records, stats))
    Path(args.stats).parent.mkdir(parents=True, exist_ok=True)
    payload = {**stats, "records": count, "output": args.output, "selection": "deterministic_round_robin"}
    Path(args.stats).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
