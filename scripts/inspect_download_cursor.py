#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

from asterlm.data.hf_stream import (
    extract_legacy_resume_plan,
    is_legacy_multistream_cursor,
    is_sequential_cursor,
)


def describe(cursor: Any) -> tuple[str, int | None, int | None, str | None]:
    if is_sequential_cursor(cursor):
        states = cursor.get("child_states")
        return (
            "asterlm-sequential-migration",
            len(states) if isinstance(states, list) else None,
            int(cursor.get("pending_cycle_skips", 0)),
            "asterlm-sequential",
        )
    if is_legacy_multistream_cursor(cursor):
        plan = extract_legacy_resume_plan(cursor)
        if plan is None:
            return "legacy-hf-multistream", None, None, None
        return "legacy-hf-multistream", plan.stream_count, plan.pending_chunks, plan.source
    if isinstance(cursor, dict):
        return "huggingface-single/bounded", None, None, None
    return type(cursor).__name__, None, None, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect an AsterLM materializer cursor without changing it")
    parser.add_argument(
        "source_dir",
        nargs="?",
        default="data/corpus-frontier-16b/fineweb_edu",
        help="Directory containing state.json and cursor-*.pkl",
    )
    args = parser.parse_args()

    root = Path(args.source_dir)
    state_path = root / "state.json"
    if not state_path.is_file():
        raise SystemExit(f"Missing {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    cursor_name = state.get("cursor_file")
    print(f"source_dir:       {root}")
    print(f"tokens:           {int(state.get('estimated_tokens', 0)):,}")
    print(f"documents_seen:   {int(state.get('documents_seen', state.get('seen', 0))):,}")
    print(f"documents_written:{int(state.get('documents_written', state.get('written', 0))):,}")
    print(f"checkpoint_id:    {state.get('checkpoint_id')}")
    print(f"checkpoint_reason:{state.get('last_checkpoint_reason')}")
    print(f"stream_layout:    {state.get('hf_stream_layout', '<legacy/unset>')}")
    print(f"cursor_file:      {cursor_name}")
    if not cursor_name:
        print("cursor_kind:      missing")
        return
    cursor_path = root / str(cursor_name)
    if not cursor_path.is_file():
        raise SystemExit(f"State references missing cursor: {cursor_path}")
    with cursor_path.open("rb") as handle:
        cursor = pickle.load(handle)
    kind, streams, pending, source = describe(cursor)
    print(f"cursor_kind:      {kind}")
    if streams is not None:
        print(f"cursor_streams:   {streams}")
    if pending is not None:
        print(f"resume_skip_chunks:{pending}")
    if source is not None:
        print(f"resume_state_from:{source}")
    if kind == "legacy-hf-multistream":
        print("next_resume:      migrate in place to one active child stream; no full restart")
    elif kind == "asterlm-sequential-migration":
        print("next_resume:      continue the already-migrated bounded sequential cursor")
    else:
        print("next_resume:      restore the saved bounded/single Hugging Face cursor")


if __name__ == "__main__":
    main()
