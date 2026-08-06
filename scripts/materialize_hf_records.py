#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from asterlm.data.hf_stream import (
    MemoryGuard,
    MemoryPressureError,
    current_rss_gib,
    memory_status,
    open_resumable_hf_stream,
    release_arrow_memory,
)
from asterlm.data.resumable import (
    RetryPolicy,
    ZstdCheckpointWriter,
    atomic_write_json,
    atomic_write_pickle,
    clean_uncommitted_outputs,
    dataset_cursor,
    is_retryable_exception,
    newest_committed_cursor,
)


@dataclass(slots=True)
class RecordSource:
    id: str
    path: str
    split: str
    # null means materialize the complete pinned split. Integer targets remain
    # useful for large post-training sources and may be increased later.
    target_records: int | None = None
    name: str | None = None
    revision: str | None = None
    keep_fields: list[str] | None = None
    columns: list[str] | None = None
    shuffle_seed: int | None = None

    def __post_init__(self) -> None:
        if self.target_records is not None and self.target_records <= 0:
            raise ValueError(f"{self.id}: target_records must be positive or null")


def legacy_signature(source: RecordSource) -> dict[str, Any]:
    """Signature used by the v2 downloader, retained only for safe migration."""
    return {
        "id": source.id,
        "path": source.path,
        "name": source.name,
        "split": source.split,
        "columns": source.columns,
        "shuffle_seed": source.shuffle_seed,
    }


def signature(source: RecordSource) -> dict[str, Any]:
    # Deliberately exclude target_records: increasing a cap must resume the same
    # source rather than force a duplicate download. Include every field that can
    # change which records/columns are written.
    return {
        "signature_version": 3,
        "id": source.id,
        "path": source.path,
        "name": source.name,
        "split": source.split,
        "revision": source.revision,
        "keep_fields": source.keep_fields,
        "columns": source.columns,
        "shuffle_seed": source.shuffle_seed,
    }


def _stored_source_compatible(stored: Any, source: RecordSource) -> bool:
    if not isinstance(stored, dict):
        return False
    expected = asdict(source)
    for key, value in expected.items():
        if key == "target_records":
            continue
        if stored.get(key) != value:
            return False
    return True


def create_base_dataset(source: RecordSource, revision: str) -> Any:
    import zstandard  # noqa: F401
    from datasets import load_dataset

    kwargs: dict[str, Any] = {
        "path": source.path,
        "name": source.name,
        "split": source.split,
        "streaming": True,
        "revision": revision,
    }
    if source.columns:
        kwargs["columns"] = source.columns
    return load_dataset(**kwargs)


def resolve_revision(source: RecordSource) -> str:
    from huggingface_hub import HfApi

    return HfApi().dataset_info(source.path, revision=source.revision).sha


def load_state(path: Path, source: RecordSource) -> dict[str, Any]:
    if not path.exists():
        return {
            "version": 3,
            "source": asdict(source),
            "source_signature": signature(source),
            "seen": 0,
            "written": 0,
            "next_shard_index": 0,
            "checkpoint_id": 0,
            "cursor_file": None,
            "retries": 0,
            "complete": False,
            "source_exhausted": False,
        }

    state = json.loads(path.read_text(encoding="utf-8"))
    expected = signature(source)
    existing = state.get("source_signature")
    if existing and existing != expected:
        # Preserve v2 checkpoints (including the user's already downloaded pilot
        # and partially materialized MMLU) when all transformation fields match.
        safely_migratable = (
            existing == legacy_signature(source)
            and _stored_source_compatible(state.get("source"), source)
        )
        if not safely_migratable:
            raise RuntimeError(
                f"Checkpoint {path} belongs to a different data transformation.\n"
                f"checkpoint={existing}\nrequested={expected}\n"
                "Move the source directory aside before changing repository, revision, "
                "split, selected columns, or retained fields."
            )
        state["signature_migrated_from"] = existing
        state["source_signature"] = expected
        state["version"] = 3

    state.setdefault("version", 3)
    state.setdefault("source_signature", expected)
    state.setdefault("source", asdict(source))
    state.setdefault("next_shard_index", len(list(path.parent.glob(f"{source.id}-*.jsonl.zst"))))
    state.setdefault("checkpoint_id", 0)
    state.setdefault("cursor_file", None)
    state.setdefault("retries", 0)
    state.setdefault("complete", False)
    state.setdefault("source_exhausted", False)
    return state


def target_reached(source: RecordSource, written: int) -> bool:
    return source.target_records is not None and written >= source.target_records


def source_complete(source: RecordSource, state: dict[str, Any]) -> bool:
    if source.target_records is None:
        return bool(state.get("source_exhausted"))
    return int(state.get("written", 0)) >= source.target_records


def commit(
    dataset: Any,
    writer: ZstdCheckpointWriter,
    state: dict[str, Any],
    state_path: Path,
    root: Path,
    *,
    complete: bool,
    reason: str,
) -> None:
    checkpoint_id = int(state.get("checkpoint_id", 0)) + 1
    cursor = dataset_cursor(dataset)
    cursor_name = None
    if cursor is not None:
        cursor_name = f"cursor-{checkpoint_id:08d}.pkl"
        atomic_write_pickle(root / cursor_name, cursor)
    shard = writer.finalize()
    updated = dict(state)
    updated.update(
        {
            "version": 3,
            "checkpoint_id": checkpoint_id,
            "cursor_file": cursor_name,
            "next_shard_index": writer.index,
            "last_shard": shard,
            "last_checkpoint_reason": reason,
            "last_checkpoint_unix": time.time(),
            "complete": complete,
        }
    )
    atomic_write_json(state_path, updated)
    state.clear()
    state.update(updated)


def materialize(
    source: RecordSource,
    root: Path,
    shard_bytes: int,
    checkpoint_seconds: float,
    checkpoint_records: int,
    retry_policy: RetryPolicy,
    max_rss_gib: float | None,
) -> dict[str, Any]:
    destination = root / source.id
    destination.mkdir(parents=True, exist_ok=True)
    state_path = destination / "state.json"
    state = load_state(state_path, source)

    # A previous capped run can have reached natural source exhaustion and then
    # failed only because the cap was too high. Switching to target_records:null
    # should recognize that complete split immediately, without re-downloading it.
    if source_complete(source, state):
        state["complete"] = True
        state["source"] = asdict(source)
        state["source_signature"] = signature(source)
        atomic_write_json(state_path, state)
        target_label = "complete split" if source.target_records is None else f"target {source.target_records:,}"
        print(f"{source.id}: already complete ({target_label}) at {int(state.get('written', 0)):,} records")
        return state

    clean_uncommitted_outputs(
        destination,
        source.id,
        int(state.get("next_shard_index", 0)),
        int(state.get("checkpoint_id", 0)),
    )
    revision = str(state.get("resolved_revision") or resolve_revision(source))
    state["resolved_revision"] = revision
    state["source"] = asdict(source)
    state["source_signature"] = signature(source)
    state.setdefault("started_at_unix", time.time())
    atomic_write_json(state_path, state)

    total = source.target_records
    initial = int(state.get("written", 0))
    progress = tqdm(
        total=total,
        initial=min(initial, total) if total is not None else initial,
        desc=source.id,
        unit="rec",
        dynamic_ncols=True,
    )
    failures = 0
    exhausted = False
    memory_guard = MemoryGuard(max_rss_gib=max_rss_gib)
    try:
        while not target_reached(source, int(state["written"])):
            cursor = newest_committed_cursor(destination, state)
            dataset, stream_layout = open_resumable_hf_stream(
                lambda: create_base_dataset(source, revision),
                seed=source.shuffle_seed,
                cursor=cursor,
                fallback_skip=int(state["seen"]) if cursor is None else 0,
                layout=state.get("hf_stream_layout"),
            )
            if state.get("hf_stream_layout") != stream_layout:
                state["hf_stream_layout"] = stream_layout
                atomic_write_json(state_path, state)
            rss = current_rss_gib()
            tqdm.write(
                f"{source.id}: stream_layout={stream_layout}; rss={rss:.2f} GiB"
                if rss is not None
                else f"{source.id}: stream_layout={stream_layout}"
            )
            writer = ZstdCheckpointWriter(
                destination,
                source.id,
                shard_bytes,
                int(state.get("next_shard_index", 0)),
            )
            since_checkpoint = 0
            try:
                for record in dataset:
                    state["seen"] = int(state["seen"]) + 1
                    since_checkpoint += 1
                    if isinstance(record, dict):
                        output = (
                            {key: record[key] for key in source.keep_fields if key in record}
                            if source.keep_fields
                            else dict(record)
                        )
                        if output:
                            output["_dataset"] = source.path
                            output["_dataset_config"] = source.name
                            output["_dataset_split"] = source.split
                            output["_resolved_revision"] = revision
                            writer.write(output)
                            state["written"] = int(state["written"]) + 1
                            progress.update(1)

                    rss, available, memory_reason = memory_guard.sample(since_checkpoint)
                    if rss is not None:
                        progress.set_postfix_str(f"{memory_status(rss, available)} layout={stream_layout}", refresh=False)
                    if memory_reason:
                        commit(
                            dataset,
                            writer,
                            state,
                            state_path,
                            destination,
                            complete=False,
                            reason="memory_pressure",
                        )
                        since_checkpoint = 0
                        raise MemoryPressureError(
                            f"{source.id}: memory guard triggered ({memory_reason}; {memory_status(rss, available)}; "
                            f"rss_limit={memory_guard.max_rss_gib:.1f}GiB; "
                            f"available_floor={memory_guard.min_available_gib:.1f}GiB). "
                            "Cursor and output were committed."
                        )

                    reached = target_reached(source, int(state["written"]))
                    due = writer.should_checkpoint(checkpoint_seconds) or (
                        checkpoint_records > 0 and since_checkpoint >= checkpoint_records
                    )
                    if reached or due:
                        commit(
                            dataset,
                            writer,
                            state,
                            state_path,
                            destination,
                            complete=reached,
                            reason="target" if reached else "periodic",
                        )
                        since_checkpoint = 0
                    if reached:
                        break
                else:
                    exhausted = True

                complete_now = (
                    exhausted
                    if source.target_records is None
                    else target_reached(source, int(state["written"]))
                )
                if writer.is_open or since_checkpoint:
                    commit(
                        dataset,
                        writer,
                        state,
                        state_path,
                        destination,
                        complete=complete_now,
                        reason="source_exhausted" if exhausted else "loop_end",
                    )
                failures = 0
                if exhausted:
                    break
            except MemoryPressureError:
                dataset = None
                release_arrow_memory()
                raise
            except KeyboardInterrupt:
                if writer.is_open or since_checkpoint:
                    commit(dataset, writer, state, state_path, destination, complete=False, reason="keyboard_interrupt")
                raise
            except BaseException as exc:
                if writer.is_open or since_checkpoint:
                    commit(
                        dataset,
                        writer,
                        state,
                        state_path,
                        destination,
                        complete=False,
                        reason=f"error:{type(exc).__name__}",
                    )
                failures += 1
                state["retries"] = int(state.get("retries", 0)) + 1
                atomic_write_json(state_path, state)
                if not is_retryable_exception(exc) or not retry_policy.permits(failures):
                    raise
                dataset = None
                release_arrow_memory()
                delay = retry_policy.delay(failures)
                tqdm.write(
                    f"{source.id}: transient {type(exc).__name__}: {exc}; "
                    f"retry {failures} in {delay:.1f}s from saved shard cursor"
                )
                time.sleep(delay)

        state["source_exhausted"] = exhausted or bool(state.get("source_exhausted"))
        state["complete"] = source_complete(source, state)
        state["finished_at_unix"] = time.time()
        state["source"] = asdict(source)
        state["source_signature"] = signature(source)
        atomic_write_json(state_path, state)
        if exhausted and not state["complete"]:
            assert source.target_records is not None
            raise RuntimeError(
                f"{source.id} ended at {int(state['written']):,} records, "
                f"below target {source.target_records:,}. Set target_records: null "
                "when the intent is to materialize the entire split."
            )
        return state
    finally:
        progress.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Interruption-safe materialization of SFT/preference HF records")
    parser.add_argument("--config", default="configs/corpus/posttrain_frontier.yaml")
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--checkpoint-seconds", type=float, default=None)
    parser.add_argument("--checkpoint-records", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=50, help="0 means unlimited transient retries")
    parser.add_argument("--retry-base-seconds", type=float, default=5.0)
    parser.add_argument("--retry-max-seconds", type=float, default=300.0)
    parser.add_argument(
        "--max-rss-gib",
        type=float,
        default=None,
        help="Commit and stop before this process exceeds the RAM limit; default is 72%% of system RAM",
    )
    args = parser.parse_args()

    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))["records"]
    root = Path(raw["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    selected = set(args.only)
    allowed = set(RecordSource.__dataclass_fields__)
    sources: list[RecordSource] = []
    for item in raw["sources"]:
        unknown = set(item) - allowed
        if unknown:
            raise ValueError(f"Unknown record-source keys: {sorted(unknown)}")
        sources.append(RecordSource(**item))
    if selected:
        available = {source.id for source in sources}
        missing = selected - available
        if missing:
            raise SystemExit(f"Unknown source IDs: {sorted(missing)}")
        sources = [source for source in sources if source.id in selected]

    retry_policy = RetryPolicy(args.max_retries, args.retry_base_seconds, args.retry_max_seconds)
    checkpoint_seconds = float(
        args.checkpoint_seconds if args.checkpoint_seconds is not None else raw.get("checkpoint_seconds", 900)
    )
    checkpoint_records = int(
        args.checkpoint_records
        if args.checkpoint_records is not None
        else raw.get("checkpoint_records", raw.get("state_flush_records", 20_000))
    )
    manifest_path = root / "manifest.json"
    manifest: dict[str, Any] = {"config": raw, "sources": {}, "updated_at_unix": time.time()}
    if manifest_path.exists():
        try:
            manifest.update(json.loads(manifest_path.read_text(encoding="utf-8")))
        except Exception:
            pass
    manifest["config"] = raw
    manifest.setdefault("sources", {})

    for source in sources:
        manifest["sources"][source.id] = materialize(
            source,
            root,
            int(raw.get("shard_size_mb", 256)) * 2**20,
            checkpoint_seconds,
            checkpoint_records,
            retry_policy,
            args.max_rss_gib,
        )
        manifest["updated_at_unix"] = time.time()
        atomic_write_json(manifest_path, manifest)
    print(f"Post-training/benchmark records are under {root}")


def _cli_entrypoint() -> None:
    try:
        main()
    except MemoryPressureError as exc:
        print(str(exc), file=sys.stderr, flush=True)
        raise SystemExit(75) from exc
    # Some pyarrow/datasets/aiohttp combinations have crashed during CPython
    # interpreter finalization *after* every shard and state file was committed.
    # A direct successful process exit avoids that teardown-only failure. Set
    # ASTERLM_MATERIALIZER_HARD_EXIT=0 for normal interpreter finalization.
    sys.stdout.flush()
    sys.stderr.flush()
    if os.environ.get("ASTERLM_MATERIALIZER_HARD_EXIT", "1") != "0":
        os._exit(0)


if __name__ == "__main__":
    _cli_entrypoint()
