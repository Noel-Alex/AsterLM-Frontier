#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import time
from pathlib import Path
from typing import Any

from asterlm.data.hf_stream import (
    LEGACY_NEXT_SHARD_LAYOUT,
    LEGACY_POLICY_NEXT_SHARD,
    convert_legacy_cursor_to_next_shard,
    convert_sequential_cursor_to_next_shard,
    is_legacy_multistream_cursor,
    is_sequential_cursor,
)
from asterlm.data.resumable import atomic_write_json, atomic_write_pickle


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def backup_path(cursor_path: Path, *, sequential: bool = False) -> Path:
    suffix = "pre-next-shard" if sequential else "legacy-multistream"
    return cursor_path.with_name(f"{cursor_path.stem}.{suffix}.pkl")


def print_plan(converted: dict[str, Any]) -> None:
    skipped = converted.get("skipped_partial_shards", [])
    print(f"policy:                  {LEGACY_POLICY_NEXT_SHARD}")
    print(f"legacy_streams:          {converted.get('legacy_stream_count')}")
    print(f"outer_pending_chunks:    {converted.get('legacy_outer_pending_chunks')}")
    print(f"resume_state_from:       {converted.get('legacy_resume_state_from')}")
    print(f"partial_files_advanced:  {len(skipped)}")
    for item in skipped:
        print(
            "  stream={stream_index:02d} partial_shard={partial_shard_index} "
            "-> next_shard={next_shard_index}; prior_rows≈{rows_already_committed_or_prefetched:,}".format(
                **item
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a legacy Hugging Face multi-stream cursor offline so AsterLM can resume "
            "at the next untouched file without rereading huge partial Parquet files."
        )
    )
    parser.add_argument(
        "source_dir",
        nargs="?",
        default="data/corpus-frontier-16b/fineweb_edu",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--restore",
        action="store_true",
        help="Restore the pre-migration cursor backup. Only use before a newer checkpoint is committed.",
    )
    args = parser.parse_args()

    root = Path(args.source_dir)
    state_path = root / "state.json"
    if not state_path.is_file():
        raise SystemExit(f"Missing {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    cursor_name = state.get("cursor_file")
    if not cursor_name:
        raise SystemExit("state.json has no cursor_file")
    cursor_path = root / str(cursor_name)
    if not cursor_path.is_file():
        raise SystemExit(f"Missing cursor {cursor_path}")
    active_cursor = load_pickle(cursor_path)
    active_is_sequential = is_sequential_cursor(active_cursor)
    backup = backup_path(cursor_path, sequential=active_is_sequential)

    if args.restore:
        migration = state.get("legacy_migration") if isinstance(state.get("legacy_migration"), dict) else {}
        recorded = migration.get("backup_cursor") if isinstance(migration, dict) else None
        candidates = [root / str(recorded)] if recorded else []
        candidates.extend([
            backup_path(cursor_path, sequential=True),
            backup_path(cursor_path, sequential=False),
        ])
        backup = next((candidate for candidate in candidates if candidate.is_file()), backup)
        if not backup.is_file():
            raise SystemExit(f"No migration backup exists for {cursor_path}")
        if args.dry_run:
            print(f"Would restore {backup} -> {cursor_path}")
            return
        temporary = cursor_path.with_name(cursor_path.name + ".restore.tmp")
        shutil.copy2(backup, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, cursor_path)
        state.pop("legacy_migration", None)
        state["hf_stream_layout"] = "legacy-multistream-sequential-v1"
        atomic_write_json(state_path, state)
        print(f"Restored original legacy cursor from {backup}")
        return

    cursor = active_cursor
    conversion_source: str
    if is_sequential_cursor(cursor):
        policy = cursor.get("migration_policy") if isinstance(cursor, dict) else None
        if policy not in (None, "", "exact", LEGACY_POLICY_NEXT_SHARD):
            raise SystemExit(f"Unsupported AsterLM sequential migration policy={policy!r}")
        converted = convert_sequential_cursor_to_next_shard(cursor)
        latest = converted.get("last_resume_skipped_partial_shards", [])
        if policy == LEGACY_POLICY_NEXT_SHARD and not latest:
            print("Cursor already uses next-shard policy and is at a clean remote shard boundary; no change made.")
            print_plan(converted)
            return
        conversion_source = (
            "asterlm-next-shard-refresh"
            if policy == LEGACY_POLICY_NEXT_SHARD
            else "asterlm-sequential-v3-v4"
        )
        backup = backup_path(cursor_path, sequential=True)
    elif is_legacy_multistream_cursor(cursor):
        converted = convert_legacy_cursor_to_next_shard(cursor)
        conversion_source = "legacy-hf-multistream"
        backup = backup_path(cursor_path, sequential=False)
    else:
        raise SystemExit("Cursor is neither a legacy Hugging Face multistream cursor nor an AsterLM sequential cursor")
    print(f"source_dir:              {root}")
    print(f"checkpoint_id:           {state.get('checkpoint_id')}")
    print(f"committed_tokens:        {int(state.get('estimated_tokens', 0)):,}")
    print(f"committed_documents:     {int(state.get('documents_seen', 0)):,}")
    print(f"conversion_source:       {conversion_source}")
    print(f"original_cursor_sha256:  {sha256(cursor_path)}")
    print_plan(converted)
    print()
    print(
        "Trade-off: all committed AsterLM shards are preserved, but the unconsumed tails of the "
        "listed partial remote files are omitted. Later untouched files from the same source replace "
        "those records toward the 54B-token target."
    )

    if args.dry_run:
        print("Dry run only; no files changed.")
        return

    if not backup.exists():
        shutil.copy2(cursor_path, backup)
        with backup.open("rb") as handle:
            os.fsync(handle.fileno())
    elif sha256(backup) != sha256(cursor_path):
        raise SystemExit(
            f"Backup already exists but differs from the active legacy cursor: {backup}. "
            "Refusing to overwrite either file."
        )

    atomic_write_pickle(cursor_path, converted)
    state["hf_stream_layout"] = LEGACY_NEXT_SHARD_LAYOUT
    state["legacy_migration"] = {
        "policy": LEGACY_POLICY_NEXT_SHARD,
        "migrated_at_unix": time.time(),
        "backup_cursor": backup.name,
        "active_cursor": cursor_path.name,
        "conversion_source": conversion_source,
        "legacy_resume_state_from": converted.get("legacy_resume_state_from"),
        "legacy_outer_pending_chunks": converted.get("legacy_outer_pending_chunks"),
        "skipped_partial_shards": converted.get("skipped_partial_shards", []),
    }
    atomic_write_json(state_path, state)

    print()
    print(f"Backup preserved at:     {backup}")
    print(f"Active cursor replaced:  {cursor_path}")
    print(f"new_cursor_sha256:       {sha256(cursor_path)}")
    print("Migration complete. No remote data was opened or downloaded.")


if __name__ == "__main__":
    main()
