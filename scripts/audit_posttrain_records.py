#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from asterlm.reasoning.io import iter_json_records

THINK_RE = re.compile(r"<\|?/?(?:think|thinking|analysis)\|?>", re.IGNORECASE)
TOOL_RE = re.compile(
    r"<\|?/?(?:tool|tool_call|tool_calls)\|?>|\"(?:commands|keystrokes|tool_calls)\"\s*:",
    re.IGNORECASE,
)
REFUSAL_RE = re.compile(
    r"\b(?:i (?:am|'m) sorry|i cannot|i can't|i am unable|as an ai|cannot assist)\b",
    re.IGNORECASE,
)
SPACE_RE = re.compile(r"\s+")


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 3)
    weight = position - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 3)


def distribution(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "min": round(min(values), 3) if values else None,
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": round(max(values), 3) if values else None,
        "mean": round(sum(values) / len(values), 3) if values else None,
    }


def normalized_text(value: Any) -> str:
    return SPACE_RE.sub(" ", str(value or "")).strip()


def message_role(message: dict[str, Any]) -> str:
    return str(message.get("role") or message.get("from") or "").strip().lower()


def message_content(message: dict[str, Any]) -> str:
    return normalized_text(message.get("content") if "content" in message else message.get("value"))


def message_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def conversation_views(record: dict[str, Any]) -> list[tuple[str, list[dict[str, Any]]]]:
    views: list[tuple[str, list[dict[str, Any]]]] = []
    for key in ("messages", "conversations"):
        messages = message_list(record.get(key))
        if messages:
            views.append((key, messages))
    for key in ("chosen", "rejected"):
        messages = message_list(record.get(key))
        if messages:
            views.append((key, messages))
    return views


def role_class(role: str) -> str:
    if role in {"assistant", "gpt", "model"}:
        return "assistant"
    if role in {"user", "human"}:
        return "user"
    if role in {"system", "tool", "function"}:
        return role
    return "unknown"


def load_token_counter(path: Path | None) -> tuple[Callable[[str], int], dict[str, Any]]:
    if path is None:
        return lambda text: len(text.split()), {"kind": "whitespace_word_proxy"}
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(path))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return (
        lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids),
        {"kind": "tokenizers_json", "path": str(path), "sha256": digest},
    )


def audit_source(
    source: Path,
    *,
    max_records: int,
    count_tokens: Callable[[str], int],
) -> dict[str, Any]:
    counters: Counter[str] = Counter()
    assistant_tokens: list[float] = []
    prompt_tokens: list[float] = []
    response_prompt_ratios: list[float] = []
    turns: list[float] = []
    unique_records: set[bytes] = set()

    for record in iter_json_records(source):
        if max_records and counters["records"] >= max_records:
            break
        counters["records"] += 1
        views = conversation_views(record)
        if not views:
            counters["records_without_conversation"] += 1
            continue

        canonical_views: list[str] = []
        record_has_assistant = False
        record_has_user = False
        for view_name, messages in views:
            counters[f"views_{view_name}"] += 1
            turns.append(float(len(messages)))
            assistant_parts: list[str] = []
            user_parts: list[str] = []
            canonical_messages: list[tuple[str, str]] = []
            for message in messages:
                role = role_class(message_role(message))
                content = message_content(message)
                canonical_messages.append((role, content))
                if role == "unknown":
                    counters["messages_unknown_role"] += 1
                if not content:
                    counters["messages_empty"] += 1
                if role == "assistant":
                    record_has_assistant = True
                    assistant_parts.append(content)
                elif role == "user":
                    record_has_user = True
                    user_parts.append(content)

            assistant = "\n".join(part for part in assistant_parts if part)
            prompt = "\n".join(part for part in user_parts if part)
            if assistant:
                assistant_count = count_tokens(assistant)
                assistant_tokens.append(float(assistant_count))
                if THINK_RE.search(assistant):
                    counters["views_with_thinking_marker"] += 1
                if TOOL_RE.search(assistant):
                    counters["views_with_tool_protocol"] += 1
                if REFUSAL_RE.search(assistant):
                    counters["views_with_refusal_phrase"] += 1
                if assistant_count >= 2048:
                    counters["views_assistant_ge_2048_tokens"] += 1
            if prompt:
                prompt_count = count_tokens(prompt)
                prompt_tokens.append(float(prompt_count))
                if assistant:
                    response_prompt_ratios.append(assistant_count / max(prompt_count, 1))
            canonical_views.append(json.dumps(canonical_messages, ensure_ascii=False, separators=(",", ":")))

        if not record_has_assistant:
            counters["records_without_assistant"] += 1
        if not record_has_user:
            counters["records_without_user"] += 1
        if record.get("result") is None and "result" in record:
            counters["records_null_result"] += 1

        fingerprint = hashlib.blake2b(
            "\n".join(canonical_views).encode("utf-8"), digest_size=16
        ).digest()
        if fingerprint in unique_records:
            counters["exact_duplicate_records"] += 1
        else:
            unique_records.add(fingerprint)

    total_records = counters["records"]
    total_views = sum(value for key, value in counters.items() if key.startswith("views_") and key in {
        "views_messages", "views_conversations", "views_chosen", "views_rejected"
    })

    def fraction(key: str, denominator: int) -> float | None:
        return round(counters[key] / denominator, 6) if denominator else None

    state_path = source / "state.json" if source.is_dir() else None
    state = None
    if state_path and state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))

    return {
        "source": str(source),
        "sampled_records": total_records,
        "materialization_state": {
            key: state.get(key)
            for key in ("seen", "written", "complete", "source_exhausted", "resolved_revision")
            if state and key in state
        },
        "counts": dict(sorted(counters.items())),
        "rates": {
            "records_without_conversation": fraction("records_without_conversation", total_records),
            "records_without_assistant": fraction("records_without_assistant", total_records),
            "records_without_user": fraction("records_without_user", total_records),
            "records_null_result": fraction("records_null_result", total_records),
            "exact_duplicate_records": fraction("exact_duplicate_records", total_records),
            "views_with_thinking_marker": fraction("views_with_thinking_marker", total_views),
            "views_with_tool_protocol": fraction("views_with_tool_protocol", total_views),
            "views_with_refusal_phrase": fraction("views_with_refusal_phrase", total_views),
            "views_assistant_ge_2048_tokens": fraction("views_assistant_ge_2048_tokens", total_views),
        },
        "distributions": {
            "assistant_tokens": distribution(assistant_tokens),
            "prompt_tokens": distribution(prompt_tokens),
            "assistant_to_prompt_token_ratio": distribution(response_prompt_ratios),
            "messages_per_view": distribution(turns),
        },
    }


def discover_sources(inputs: list[str]) -> list[Path]:
    sources: list[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir() and (path / "manifest.json").is_file():
            sources.extend(sorted(item for item in path.iterdir() if item.is_dir()))
        else:
            sources.append(path)
    return sources


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit post-training records without promoting them")
    parser.add_argument("inputs", nargs="+", help="Materialization roots, source directories, or record files")
    parser.add_argument("--max-records", type=int, default=10000, help="Deterministic prefix per source; 0 scans all")
    parser.add_argument("--tokenizer", type=Path, help="Optional tokenizers JSON for exact token counts")
    parser.add_argument("--output", type=Path, default=Path("data/audits/posttrain_content_audit.json"))
    args = parser.parse_args()

    count_tokens, tokenizer_metadata = load_token_counter(args.tokenizer)
    started = time.time()
    reports = [
        audit_source(source, max_records=args.max_records, count_tokens=count_tokens)
        for source in discover_sources(args.inputs)
    ]
    payload = {
        "schema_version": 1,
        "created_at_unix": time.time(),
        "elapsed_seconds": round(time.time() - started, 3),
        "selection": "materialized_deterministic_prefix_after_source_shuffle",
        "max_records_per_source": args.max_records,
        "token_counter": tokenizer_metadata,
        "sources": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
