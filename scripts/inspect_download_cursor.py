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


def describe(cursor: Any) -> tuple[str, int | None, int | None, str | None, str | None]:
    if is_sequential_cursor(cursor):
        states = cursor.get("child_states")
        raw_policy = cursor.get("migration_policy")
        policy = str(raw_policy) if raw_policy not in (None, "") else "exact/unset"
        kind = "asterlm-next-shard-migration" if policy == "next-shard" else "asterlm-sequential-migration"
        return (
            kind,
            len(states) if isinstance(states, list) else None,
            int(cursor.get("pending_cycle_skips", 0)),
            "asterlm-sequential",
            policy,
        )
    if is_legacy_multistream_cursor(cursor):
        plan = extract_legacy_resume_plan(cursor)
        if plan is None:
            return "legacy-hf-multistream", None, None, None, "next-shard"
        return "legacy-hf-multistream", plan.stream_count, plan.pending_chunks, plan.source, "next-shard"
    if isinstance(cursor, dict):
        return "huggingface-single/bounded", None, None, None, None
    return type(cursor).__name__, None, None, None, None


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
    kind, streams, pending, source, policy = describe(cursor)
    print(f"cursor_kind:      {kind}")
    if streams is not None:
        print(f"cursor_streams:   {streams}")
    if pending is not None:
        print(f"resume_skip_chunks:{pending}")
    if source is not None:
        print(f"resume_state_from:{source}")
    if policy is not None:
        print(f"migration_policy: {policy}")
    if kind == "legacy-hf-multistream":
        print("next_resume:      offline next-shard migration recommended; preserves committed output")
        print("tradeoff:         omits only the unconsumed tails of the 10 partial remote files")
    elif kind == "asterlm-next-shard-migration":
        skipped = cursor.get("skipped_partial_shards", []) if isinstance(cursor, dict) else []
        latest = cursor.get("last_resume_skipped_partial_shards", []) if isinstance(cursor, dict) else []
        print(f"partial_files_skipped_total:{len(skipped) if isinstance(skipped, list) else 0}")
        print(f"partial_files_skipped_last_resume:{len(latest) if isinstance(latest, list) else 0}")
        print("restart_policy:   automatically advance any newly-partial legacy file on every restart")
        print("next_resume:      continue from an untouched remote file; no multi-GB row replay")
    elif kind == "asterlm-sequential-migration":
        print("next_resume:      run migrate_legacy_cursor.py offline; this exact cursor may reread large partial files")
    else:
        print("next_resume:      restore the saved bounded/single Hugging Face cursor")


if __name__ == "__main__":
    main()
