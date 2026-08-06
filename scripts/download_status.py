#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def state_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("data/**/state.json")):
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            rows.append({"path": str(path), "error": str(exc)})
            continue
        source_state = state.get("source", {})
        target_tokens = state.get("target_tokens") or source_state.get("target_tokens")
        has_record_counter = "written" in state or "target_records" in source_state
        target_records = source_state.get("target_records") if has_record_counter else None
        current = state.get("estimated_tokens") if target_tokens is not None else state.get("written")
        target = target_tokens if target_tokens is not None else target_records
        pct = (100.0 * float(current) / float(target)) if current is not None and target else None
        rows.append(
            {
                "path": str(path),
                "source": state.get("source", {}).get("id") or path.parent.name,
                "current": current,
                "target": target,
                "percent": pct,
                "seen": state.get("documents_seen", state.get("seen")),
                "written": state.get("documents_written", state.get("written")),
                "shards": state.get("next_shard_index"),
                "retries": state.get("retries", 0),
                "complete": state.get("complete", False),
                "revision": state.get("resolved_revision"),
                "last_reason": state.get("last_checkpoint_reason"),
            }
        )
    return rows


def human(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        for suffix, scale in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
            if abs(value) >= scale:
                return f"{value / scale:.2f}{suffix}"
        return f"{value:,.0f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize AsterLM data-download checkpoints")
    parser.add_argument("--root", default=".")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    rows = state_rows(Path(args.root))
    if args.json:
        print(json.dumps(rows, indent=2, default=str))
        return
    if not rows:
        print("No data/**/state.json checkpoints found.")
        return
    print(f"{'SOURCE':34} {'PROGRESS':>19} {'PCT':>8} {'SHARDS':>7} {'RETRY':>6} {'DONE':>5}  LAST")
    for row in rows:
        source = str(row.get("source", "-"))
        if len(source) > 34:
            source = source[:31] + "..."
        pct = f"{row['percent']:.2f}%" if row.get("percent") is not None else "-"
        target = row.get("target")
        progress = (
            f"{human(row.get('current'))}/{human(target)}"
            if target is not None
            else f"{human(row.get('current'))}/ALL"
        )
        print(
            f"{source:34} {progress:>19} {pct:>8} {str(row.get('shards', '-')):>7} "
            f"{str(row.get('retries', 0)):>6} {str(bool(row.get('complete'))):>5}  {row.get('last_reason') or '-'}"
        )


if __name__ == "__main__":
    main()
