#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
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
class CorpusSource:
    id: str
    path: str
    target_tokens: int
    name: str | None = None
    split: str = "train"
    text_field: str = "text"
    revision: str | None = None
    token_count_field: str | None = None
    min_chars: int = 128
    max_chars: int = 500_000
    require_fields: dict[str, Any] | None = None
    columns: list[str] | None = None
    shuffle_seed: int | None = None


def nested_get(record: dict[str, Any], field: str) -> Any:
    value: Any = record
    for part in field.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def estimate_tokens(text: str, record: dict[str, Any], field: str | None) -> int:
    if field:
        value = nested_get(record, field)
        if isinstance(value, (int, float)) and value > 0:
            return int(value)
    return max(1, round(len(text.encode("utf-8")) / 4.0))


def resolve_revision(repo_id: str, requested: str | None) -> str:
    from huggingface_hub import HfApi

    return HfApi().dataset_info(repo_id, revision=requested).sha


def create_base_dataset(source: CorpusSource, revision: str) -> Any:
    # Importing zstandard before fsspec/datasets ensures .zst support is registered.
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


def legacy_source_signature(source: CorpusSource) -> dict[str, Any]:
    """Signature used by the v2 downloader, retained only for safe migration."""
    return {
        "id": source.id,
        "path": source.path,
        "name": source.name,
        "split": source.split,
        "text_field": source.text_field,
        "columns": source.columns,
        "shuffle_seed": source.shuffle_seed,
    }


def source_signature(source: CorpusSource) -> dict[str, Any]:
    # Deliberately exclude target_tokens so the pilot can expand in-place into
    # the frontier corpus. Include every field that changes accepted text.
    return {
        "signature_version": 3,
        "id": source.id,
        "path": source.path,
        "name": source.name,
        "split": source.split,
        "text_field": source.text_field,
        "revision": source.revision,
        "token_count_field": source.token_count_field,
        "min_chars": source.min_chars,
        "max_chars": source.max_chars,
        "require_fields": source.require_fields,
        "columns": source.columns,
        "shuffle_seed": source.shuffle_seed,
    }


def _stored_source_compatible(stored: Any, source: CorpusSource) -> bool:
    if not isinstance(stored, dict):
        return False
    expected = asdict(source)
    for key, value in expected.items():
        if key == "target_tokens":
            continue
        if stored.get(key) != value:
            return False
    return True


def default_state(source: CorpusSource) -> dict[str, Any]:
    return {
        "version": 3,
        "source": asdict(source),
        "source_signature": source_signature(source),
        "documents_seen": 0,
        "documents_written": 0,
        "estimated_tokens": 0,
        "bytes": 0,
        "next_shard_index": 0,
        "checkpoint_id": 0,
        "cursor_file": None,
        "retries": 0,
        "complete": False,
    }


def load_state(path: Path, source: CorpusSource) -> dict[str, Any]:
    if not path.exists():
        return default_state(source)
    state = json.loads(path.read_text(encoding="utf-8"))
    expected = source_signature(source)
    existing = state.get("source_signature")
    if existing and existing != expected:
        safely_migratable = (
            existing == legacy_source_signature(source)
            and _stored_source_compatible(state.get("source"), source)
        )
        if not safely_migratable:
            raise RuntimeError(
                f"Existing checkpoint at {path} belongs to a different data transformation.\n"
                f"checkpoint={existing}\nrequested={expected}\n"
                "Move the source directory aside before changing repository, revision, split, "
                "text field, filters, selected columns, or required fields."
            )
        state["signature_migrated_from"] = existing
        state["source_signature"] = expected
        state["version"] = 3
    # Migrate v1 record-count checkpoints. The first v2/v3 resume may use .skip(), after
    # which all subsequent checkpoints use the dataset's shard-aware state_dict.
    state.setdefault("version", 1)
    state.setdefault("source", asdict(source))
    state.setdefault("source_signature", expected)
    state.setdefault("next_shard_index", len(list(path.parent.glob(f"{source.id}-*.jsonl.zst"))))
    state.setdefault("checkpoint_id", 0)
    state.setdefault("cursor_file", None)
    state.setdefault("retries", 0)
    state.setdefault("complete", False)
    return state


def matches_requirements(record: dict[str, Any], required: dict[str, Any] | None) -> bool:
    if not required:
        return True
    return all(nested_get(record, key) == expected for key, expected in required.items())


def commit_checkpoint(
    *,
    dataset: Any,
    writer: ZstdCheckpointWriter,
    state: dict[str, Any],
    state_path: Path,
    root: Path,
    complete: bool | None = None,
    reason: str,
) -> None:
    """Commit cursor + optional shard, advancing state.json last.

    state.json is the atomic pointer. If power is lost before it is replaced, the
    next run deletes the orphan cursor/shard and replays at most one checkpoint.
    """

    checkpoint_id = int(state.get("checkpoint_id", 0)) + 1
    cursor = dataset_cursor(dataset)
    cursor_name: str | None = None
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
            "last_checkpoint_unix": time.time(),
            "last_checkpoint_reason": reason,
            "last_shard": shard,
        }
    )
    if complete is not None:
        updated["complete"] = complete
    atomic_write_json(state_path, updated)
    state.clear()
    state.update(updated)


def materialize(
    source: CorpusSource,
    output_root: Path,
    shard_bytes: int,
    checkpoint_seconds: float,
    checkpoint_documents: int,
    retry_policy: RetryPolicy,
    max_rss_gib: float | None,
) -> dict[str, Any]:
    source_root = output_root / source.id
    source_root.mkdir(parents=True, exist_ok=True)
    state_path = source_root / "state.json"
    state = load_state(state_path, source)

    if state.get("complete") and int(state.get("estimated_tokens", 0)) >= source.target_tokens:
        print(f"{source.id}: already complete at approximately {int(state['estimated_tokens']):,} tokens")
        return state

    clean_uncommitted_outputs(
        source_root,
        source.id,
        int(state.get("next_shard_index", 0)),
        int(state.get("checkpoint_id", 0)),
    )

    revision = str(state.get("resolved_revision") or resolve_revision(source.path, source.revision))
    if source.revision and revision != source.revision:
        # Requested branches/tags resolve to a commit. Explicit commit hashes match directly.
        pass
    state["resolved_revision"] = revision
    state["source"] = asdict(source)
    state["source_signature"] = source_signature(source)
    state.setdefault("started_at_unix", time.time())
    atomic_write_json(state_path, state)

    progress = tqdm(
        total=source.target_tokens,
        initial=min(int(state["estimated_tokens"]), source.target_tokens),
        unit="tok",
        unit_scale=True,
        desc=source.id,
        dynamic_ncols=True,
    )
    failures = 0
    exhausted = False
    memory_guard = MemoryGuard(max_rss_gib=max_rss_gib)

    try:
        while int(state["estimated_tokens"]) < source.target_tokens:
            cursor = newest_committed_cursor(source_root, state)
            dataset, stream_layout = open_resumable_hf_stream(
                lambda: create_base_dataset(source, revision),
                seed=source.shuffle_seed,
                cursor=cursor,
                fallback_skip=int(state["documents_seen"]) if cursor is None else 0,
                layout=state.get("hf_stream_layout"),
            )
            if state.get("hf_stream_layout") != stream_layout:
                state["hf_stream_layout"] = stream_layout
                atomic_write_json(state_path, state)
            rss = current_rss_gib()
            tqdm.write(
                f"{source.id}: stream_layout={stream_layout}; "
                f"rss={rss:.2f} GiB" if rss is not None else f"{source.id}: stream_layout={stream_layout}"
            )
            writer = ZstdCheckpointWriter(
                source_root,
                source.id,
                shard_bytes,
                int(state.get("next_shard_index", 0)),
            )
            records_since_checkpoint = 0
            try:
                for record in dataset:
                    state["documents_seen"] = int(state["documents_seen"]) + 1
                    records_since_checkpoint += 1
                    if isinstance(record, dict) and matches_requirements(record, source.require_fields):
                        value = nested_get(record, source.text_field)
                        if isinstance(value, str):
                            text = value.strip()
                            if source.min_chars <= len(text) <= source.max_chars and "\x00" not in text:
                                tokens = estimate_tokens(text, record, source.token_count_field)
                                doc_id = record.get("id") or record.get("blob_id") or hashlib.blake2b(
                                    text[:8192].encode("utf-8", errors="ignore"), digest_size=16
                                ).hexdigest()
                                output = {
                                    "text": text,
                                    "source": source.id,
                                    "source_id": str(doc_id),
                                    "estimated_tokens": tokens,
                                }
                                for key in (
                                    "url",
                                    "language",
                                    "language_score",
                                    "score",
                                    "int_score",
                                    "fasttext_score",
                                    "license_type",
                                    "detected_licenses",
                                ):
                                    if key in record:
                                        output[key] = record[key]
                                writer.write(output)
                                state["documents_written"] = int(state["documents_written"]) + 1
                                state["estimated_tokens"] = int(state["estimated_tokens"]) + tokens
                                state["bytes"] = int(state["bytes"]) + len(text.encode("utf-8", errors="ignore"))
                                progress.update(tokens)

                    rss, available, memory_reason = memory_guard.sample(records_since_checkpoint)
                    if rss is not None:
                        progress.set_postfix_str(
                            f"{memory_status(rss, available)} layout={stream_layout}", refresh=False
                        )
                    if memory_reason:
                        commit_checkpoint(
                            dataset=dataset,
                            writer=writer,
                            state=state,
                            state_path=state_path,
                            root=source_root,
                            complete=False,
                            reason="memory_pressure",
                        )
                        records_since_checkpoint = 0
                        raise MemoryPressureError(
                            f"{source.id}: memory guard triggered ({memory_reason}; {memory_status(rss, available)}; "
                            f"rss_limit={memory_guard.max_rss_gib:.1f}GiB; "
                            f"available_floor={memory_guard.min_available_gib:.1f}GiB). "
                            "Cursor and output were committed. "
                            "Do not disable this guard; investigate the current upstream row group."
                        )

                    target_reached = int(state["estimated_tokens"]) >= source.target_tokens
                    checkpoint_due = writer.should_checkpoint(checkpoint_seconds) or (
                        checkpoint_documents > 0 and records_since_checkpoint >= checkpoint_documents
                    )
                    if target_reached or checkpoint_due:
                        commit_checkpoint(
                            dataset=dataset,
                            writer=writer,
                            state=state,
                            state_path=state_path,
                            root=source_root,
                            complete=target_reached,
                            reason="target" if target_reached else "periodic",
                        )
                        records_since_checkpoint = 0
                    if target_reached:
                        break
                else:
                    exhausted = True

                if writer.is_open or records_since_checkpoint:
                    commit_checkpoint(
                        dataset=dataset,
                        writer=writer,
                        state=state,
                        state_path=state_path,
                        root=source_root,
                        complete=int(state["estimated_tokens"]) >= source.target_tokens,
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
                if writer.is_open or records_since_checkpoint:
                    commit_checkpoint(
                        dataset=dataset,
                        writer=writer,
                        state=state,
                        state_path=state_path,
                        root=source_root,
                        complete=False,
                        reason="keyboard_interrupt",
                    )
                raise
            except BaseException as exc:
                if writer.is_open or records_since_checkpoint:
                    commit_checkpoint(
                        dataset=dataset,
                        writer=writer,
                        state=state,
                        state_path=state_path,
                        root=source_root,
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

        state["complete"] = int(state["estimated_tokens"]) >= source.target_tokens
        state["source_exhausted"] = exhausted
        state["finished_at_unix"] = time.time()
        atomic_write_json(state_path, state)
        if exhausted and not state["complete"]:
            raise RuntimeError(
                f"{source.id} ended at approximately {int(state['estimated_tokens']):,} tokens, "
                f"below target {source.target_tokens:,}."
            )
        return state
    finally:
        progress.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize a revision-pinned, interruption-safe LLM corpus")
    parser.add_argument("--config", required=True)
    parser.add_argument("--only", action="append", default=[], help="Materialize only these source IDs")
    parser.add_argument("--checkpoint-seconds", type=float, default=None)
    parser.add_argument("--checkpoint-documents", type=int, default=None)
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

    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))["corpus"]
    output = Path(raw.get("output_dir", "data/corpus"))
    output.mkdir(parents=True, exist_ok=True)
    shard_bytes = int(raw.get("shard_size_mb", 512)) * 2**20
    checkpoint_seconds = float(
        args.checkpoint_seconds if args.checkpoint_seconds is not None else raw.get("checkpoint_seconds", 900)
    )
    checkpoint_documents = int(
        args.checkpoint_documents
        if args.checkpoint_documents is not None
        else raw.get("checkpoint_documents", raw.get("state_flush_documents", 20_000))
    )
    retry_policy = RetryPolicy(
        max_retries=args.max_retries,
        base_seconds=args.retry_base_seconds,
        max_seconds=args.retry_max_seconds,
    )

    allowed = set(CorpusSource.__dataclass_fields__)
    sources: list[CorpusSource] = []
    for item in raw["sources"]:
        unknown = set(item) - allowed
        if unknown:
            raise ValueError(f"Unknown corpus source keys: {sorted(unknown)}")
        sources.append(CorpusSource(**item))

    if args.only:
        selected = set(args.only)
        available = {source.id for source in sources}
        missing = selected - available
        if missing:
            raise SystemExit(f"Unknown source IDs: {sorted(missing)}")
        sources = [source for source in sources if source.id in selected]

    manifest_path = output / "manifest.json"
    manifest: dict[str, Any] = {
        "config_file": args.config,
        "config": raw,
        "sources": {},
        "updated_at_unix": time.time(),
    }
    if manifest_path.exists():
        try:
            manifest.update(json.loads(manifest_path.read_text(encoding="utf-8")))
        except Exception:
            pass
    manifest["config_file"] = args.config
    manifest["config"] = raw
    manifest.setdefault("sources", {})

    for source in sources:
        manifest["sources"][source.id] = materialize(
            source,
            output,
            shard_bytes,
            checkpoint_seconds,
            checkpoint_documents,
            retry_policy,
            args.max_rss_gib,
        )
        manifest["updated_at_unix"] = time.time()
        atomic_write_json(manifest_path, manifest)

    total = sum(int(item.get("estimated_tokens", 0)) for item in manifest["sources"].values())
    print(f"Materialized approximately {total:,} tokens under {output}")


def _cli_entrypoint() -> None:
    try:
        main()
    except MemoryPressureError as exc:
        print(str(exc), file=sys.stderr, flush=True)
        raise SystemExit(75) from exc
    # Avoid a known datasets/pyarrow background-thread crash during CPython
    # finalization after all committed output has already been written.
    sys.stdout.flush()
    sys.stderr.flush()
    if os.environ.get("ASTERLM_MATERIALIZER_HARD_EXIT", "1") != "0":
        os._exit(0)


if __name__ == "__main__":
    _cli_entrypoint()
