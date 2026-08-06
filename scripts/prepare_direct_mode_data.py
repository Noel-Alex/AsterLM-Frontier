#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

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


def converted(inputs: list[str], max_records: int) -> Iterator[dict[str, Any]]:
    emitted = 0
    seen: set[str] = set()
    for input_path in inputs:
        for record in iter_json_records(input_path):
            messages = messages_from_record(record)
            if not messages:
                continue
            rendered = json.dumps(messages, sort_keys=True, ensure_ascii=False)
            key = stable_id(rendered)
            if key in seen:
                continue
            seen.add(key)
            yield {
                "id": key,
                "messages": direct_messages(messages),
                "source": str(record.get("source", record.get("_dataset", "unknown"))),
                "mode": "direct",
            }
            emitted += 1
            if max_records and emitted >= max_records:
                return


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert broad instruction records to Aster direct-mode SFT")
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", default="data/reasoning/direct_mode_sft.jsonl")
    parser.add_argument("--max-records", type=int, default=150000)
    parser.add_argument("--stats", default="data/reasoning/direct_mode_stats.json")
    args = parser.parse_args()
    count = atomic_write_jsonl(args.output, converted(args.inputs, args.max_records))
    Path(args.stats).parent.mkdir(parents=True, exist_ok=True)
    Path(args.stats).write_text(json.dumps({"records": count, "output": args.output}, indent=2), encoding="utf-8")
    print(json.dumps({"records": count, "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
