#!/usr/bin/env python
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
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

_thread_local = threading.local()
_s3_settings: dict[str, Any] = {}


@dataclass(slots=True)
class FetchResult:
    item: dict[str, Any] | None
    permanent_skip: bool = False
    error: str | None = None


def s3_client() -> Any:
    client = getattr(_thread_local, "s3", None)
    if client is None:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config

        client = boto3.client(
            "s3",
            config=Config(
                signature_version=UNSIGNED,
                connect_timeout=int(_s3_settings.get("connect_timeout", 60)),
                read_timeout=int(_s3_settings.get("read_timeout", 300)),
                max_pool_connections=int(_s3_settings.get("max_pool_connections", 16)),
                retries={"max_attempts": int(_s3_settings.get("sdk_retries", 20)), "mode": "adaptive"},
            ),
        )
        _thread_local.s3 = client
    return client


def fetch(record: dict[str, Any], policy: RetryPolicy) -> FetchResult:
    if record.get("license_type") != "permissive":
        return FetchResult(None, permanent_skip=True, error="non_permissive")

    from botocore.exceptions import ClientError

    failures = 0
    while True:
        try:
            obj = s3_client().get_object(Bucket="softwareheritage", Key=f"content/{record['blob_id']}")
            with gzip.GzipFile(fileobj=obj["Body"]) as fin:
                raw = fin.read()
            encoding = str(record.get("src_encoding") or "utf-8")
            try:
                text = raw.decode(encoding, errors="replace")
            except LookupError:
                text = raw.decode("utf-8", errors="replace")
            text = text.strip()
            if len(text) < 64 or "\x00" in text:
                return FetchResult(None, permanent_skip=True, error="invalid_text")
            return FetchResult(
                {
                    "text": text,
                    "source": "stack_edu",
                    "source_id": str(record["blob_id"]),
                    "language": record.get("language"),
                    "repo_name": record.get("repo_name"),
                    "path": record.get("path"),
                    "license_type": record.get("license_type"),
                    "detected_licenses": record.get("detected_licenses"),
                    "score": record.get("score"),
                    "estimated_tokens": max(1, round(len(raw) / 4.0)),
                }
            )
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            status = int(exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0) or 0)
            if code in {"NoSuchKey", "404"} or status == 404:
                return FetchResult(None, permanent_skip=True, error=f"{code or status}: {exc}")
            if code in {"AccessDenied", "InvalidAccessKeyId"} or status in {400, 401, 403}:
                raise
            failures += 1
            if not policy.permits(failures):
                raise
            time.sleep(policy.delay(failures))
        except BaseException as exc:
            failures += 1
            if not is_retryable_exception(exc) or not policy.permits(failures):
                raise
            time.sleep(policy.delay(failures))


def resolve_revision(requested: str | None) -> str:
    from huggingface_hub import HfApi

    return HfApi().dataset_info("HuggingFaceTB/stack-edu", revision=requested).sha


def stable_seed(language: str) -> int:
    return 3000 + sum((index + 1) * ord(char) for index, char in enumerate(language)) % 100000


def create_base_dataset(language: str, revision: str) -> Any:
    import zstandard  # noqa: F401
    from datasets import load_dataset

    return load_dataset(
        "HuggingFaceTB/stack-edu",
        language,
        split="train",
        streaming=True,
        revision=revision,
    )


def load_state(path: Path, language: str, target_tokens: int) -> dict[str, Any]:
    signature = {
        "repo": "HuggingFaceTB/stack-edu",
        "language": language,
        "split": "train",
        "shuffle_seed": stable_seed(language),
    }
    if not path.exists():
        return {
            "version": 2,
            "source_signature": signature,
            "target_tokens": target_tokens,
            "documents_seen": 0,
            "documents_written": 0,
            "estimated_tokens": 0,
            "permanent_skips": 0,
            "fetch_errors": 0,
            "next_shard_index": 0,
            "checkpoint_id": 0,
            "cursor_file": None,
            "retries": 0,
            "complete": False,
        }
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("source_signature") and state["source_signature"] != signature:
        raise RuntimeError(f"Checkpoint {path} belongs to {state['source_signature']}, not {signature}")
    state.setdefault("source_signature", signature)
    state.setdefault("next_shard_index", len(list(path.parent.glob(f"{language.lower()}-*.jsonl.zst"))))
    state.setdefault("checkpoint_id", 0)
    state.setdefault("cursor_file", None)
    state.setdefault("retries", 0)
    state.setdefault("permanent_skips", int(state.get("failed", 0)))
    state.setdefault("fetch_errors", 0)
    state.setdefault("complete", False)
    return state


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
            "version": 2,
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


def consume_done(
    done: set[Future[FetchResult]],
    writer: ZstdCheckpointWriter,
    state: dict[str, Any],
    progress: tqdm,
) -> None:
    for future in done:
        result = future.result()
        if result.item is None:
            if result.permanent_skip:
                state["permanent_skips"] = int(state.get("permanent_skips", 0)) + 1
            else:
                state["fetch_errors"] = int(state.get("fetch_errors", 0)) + 1
            continue
        writer.write(result.item)
        tokens = int(result.item["estimated_tokens"])
        state["documents_written"] = int(state["documents_written"]) + 1
        state["estimated_tokens"] = int(state["estimated_tokens"]) + tokens
        progress.update(tokens)


def drain_pending(
    pending: set[Future[FetchResult]],
    writer: ZstdCheckpointWriter,
    state: dict[str, Any],
    progress: tqdm,
    *,
    stop_at_target: int | None = None,
) -> set[Future[FetchResult]]:
    while pending:
        done, pending = wait(pending, return_when=FIRST_COMPLETED)
        consume_done(done, writer, state, progress)
        if stop_at_target is not None and int(state["estimated_tokens"]) >= stop_at_target:
            # Already-started requests are allowed to finish before a cursor checkpoint;
            # this avoids skipping submitted metadata records on resume.
            continue
    return pending


def materialize_language(
    language: str,
    target_tokens: int,
    output: Path,
    workers: int,
    in_flight: int,
    shard_bytes: int,
    revision: str | None,
    checkpoint_seconds: float,
    checkpoint_records: int,
    retry_policy: RetryPolicy,
    max_rss_gib: float | None,
) -> dict[str, Any]:
    prefix = language.lower().replace("-", "_")
    root = output / prefix
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "state.json"
    state = load_state(state_path, language, target_tokens)
    if state.get("complete") and int(state.get("estimated_tokens", 0)) >= target_tokens:
        print(f"{language}: already complete at approximately {int(state['estimated_tokens']):,} tokens")
        return state

    clean_uncommitted_outputs(
        root,
        prefix,
        int(state.get("next_shard_index", 0)),
        int(state.get("checkpoint_id", 0)),
    )
    resolved = str(state.get("resolved_revision") or resolve_revision(revision))
    state["resolved_revision"] = resolved
    state["target_tokens"] = target_tokens
    state.setdefault("started_at_unix", time.time())
    atomic_write_json(state_path, state)

    progress = tqdm(
        total=target_tokens,
        initial=min(int(state["estimated_tokens"]), target_tokens),
        unit="tok",
        unit_scale=True,
        desc=language,
        dynamic_ncols=True,
    )
    failures = 0
    exhausted = False
    memory_guard = MemoryGuard(max_rss_gib=max_rss_gib)
    try:
        while int(state["estimated_tokens"]) < target_tokens:
            cursor = newest_committed_cursor(root, state)
            dataset, stream_layout = open_resumable_hf_stream(
                lambda: create_base_dataset(language, resolved),
                seed=stable_seed(language),
                cursor=cursor,
                fallback_skip=int(state["documents_seen"]) if cursor is None else 0,
                layout=state.get("hf_stream_layout"),
            )
            if state.get("hf_stream_layout") != stream_layout:
                state["hf_stream_layout"] = stream_layout
                atomic_write_json(state_path, state)
            rss = current_rss_gib()
            tqdm.write(
                f"{language}: stream_layout={stream_layout}; rss={rss:.2f} GiB"
                if rss is not None
                else f"{language}: stream_layout={stream_layout}"
            )
            writer = ZstdCheckpointWriter(root, prefix, shard_bytes, int(state.get("next_shard_index", 0)))
            pending: set[Future[FetchResult]] = set()
            records_since_checkpoint = 0
            checkpoint_started = time.monotonic()
            try:
                with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"stack-{prefix}") as pool:
                    for record in dataset:
                        if int(state["estimated_tokens"]) >= target_tokens:
                            break
                        state["documents_seen"] = int(state["documents_seen"]) + 1
                        records_since_checkpoint += 1
                        if isinstance(record, dict) and record.get("license_type") == "permissive":
                            pending.add(pool.submit(fetch, dict(record), retry_policy))
                        if len(pending) >= in_flight:
                            done, pending = wait(pending, return_when=FIRST_COMPLETED)
                            consume_done(done, writer, state, progress)

                        rss, available, memory_reason = memory_guard.sample(records_since_checkpoint)
                        if rss is not None:
                            progress.set_postfix_str(
                                f"{memory_status(rss, available)} layout={stream_layout} pending={len(pending)}",
                                refresh=False,
                            )
                        if memory_reason:
                            pending = drain_pending(pending, writer, state, progress)
                            commit(
                                dataset,
                                writer,
                                state,
                                state_path,
                                root,
                                complete=False,
                                reason="memory_pressure",
                            )
                            records_since_checkpoint = 0
                            raise MemoryPressureError(
                                f"{language}: memory guard triggered ({memory_reason}; {memory_status(rss, available)}; "
                                f"rss_limit={memory_guard.max_rss_gib:.1f}GiB; "
                                f"available_floor={memory_guard.min_available_gib:.1f}GiB). "
                                "Cursor and output were committed."
                            )

                        due = (
                            writer.should_checkpoint(checkpoint_seconds)
                            or (checkpoint_records > 0 and records_since_checkpoint >= checkpoint_records)
                            or (
                                checkpoint_seconds > 0
                                and time.monotonic() - checkpoint_started >= checkpoint_seconds
                            )
                        )
                        if due or int(state["estimated_tokens"]) >= target_tokens:
                            pending = drain_pending(pending, writer, state, progress)
                            complete = int(state["estimated_tokens"]) >= target_tokens
                            commit(
                                dataset,
                                writer,
                                state,
                                state_path,
                                root,
                                complete=complete,
                                reason="target" if complete else "periodic",
                            )
                            records_since_checkpoint = 0
                            checkpoint_started = time.monotonic()
                            if complete:
                                break
                    else:
                        exhausted = True

                    pending = drain_pending(pending, writer, state, progress)
                    if writer.is_open or records_since_checkpoint:
                        commit(
                            dataset,
                            writer,
                            state,
                            state_path,
                            root,
                            complete=int(state["estimated_tokens"]) >= target_tokens,
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
                # Executor context waits for already-running requests. Drain completed
                # outputs before committing the corresponding HF cursor.
                pending = drain_pending(pending, writer, state, progress)
                if writer.is_open or records_since_checkpoint:
                    commit(dataset, writer, state, state_path, root, complete=False, reason="keyboard_interrupt")
                raise
            except BaseException as exc:
                pending = drain_pending(pending, writer, state, progress)
                if writer.is_open or records_since_checkpoint:
                    commit(
                        dataset,
                        writer,
                        state,
                        state_path,
                        root,
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
                    f"{language}: transient {type(exc).__name__}: {exc}; "
                    f"retry {failures} in {delay:.1f}s from saved shard cursor"
                )
                time.sleep(delay)

        state["complete"] = int(state["estimated_tokens"]) >= target_tokens
        state["source_exhausted"] = exhausted
        state["finished_at_unix"] = time.time()
        atomic_write_json(state_path, state)
        if exhausted and not state["complete"]:
            raise RuntimeError(
                f"{language} ended at approximately {int(state['estimated_tokens']):,} tokens, "
                f"below target {target_tokens:,}."
            )
        return state
    finally:
        progress.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Interruption-safe permissive multilingual Stack-Edu materializer")
    parser.add_argument("--config", default="configs/corpus/stack_edu_2p4b.yaml")
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--revision", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--in-flight", type=int, default=None)
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

    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))["stack_edu"]
    output = Path(raw["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    selected = set(args.only)
    workers = int(args.workers if args.workers is not None else raw.get("workers", 4))
    in_flight = int(args.in_flight if args.in_flight is not None else raw.get("in_flight", workers * 4))
    if workers < 1 or in_flight < workers:
        raise SystemExit("workers must be >=1 and in-flight must be >= workers")

    _s3_settings.update(
        {
            "connect_timeout": int(raw.get("connect_timeout_seconds", 60)),
            "read_timeout": int(raw.get("read_timeout_seconds", 300)),
            "max_pool_connections": max(in_flight, workers),
            "sdk_retries": int(raw.get("sdk_retries", 20)),
        }
    )
    checkpoint_seconds = float(
        args.checkpoint_seconds if args.checkpoint_seconds is not None else raw.get("checkpoint_seconds", 900)
    )
    checkpoint_records = int(
        args.checkpoint_records if args.checkpoint_records is not None else raw.get("checkpoint_records", 10_000)
    )
    retry_policy = RetryPolicy(args.max_retries, args.retry_base_seconds, args.retry_max_seconds)

    manifest_path = output / "manifest.json"
    manifest: dict[str, Any] = {"languages": {}, "updated_at_unix": time.time(), "config": raw}
    if manifest_path.exists():
        try:
            manifest.update(json.loads(manifest_path.read_text(encoding="utf-8")))
        except Exception:
            pass
    manifest.setdefault("languages", {})
    manifest["config"] = raw

    for item in raw["languages"]:
        if selected and item["name"] not in selected:
            continue
        manifest["languages"][item["name"]] = materialize_language(
            item["name"],
            int(item["target_tokens"]),
            output,
            workers,
            in_flight,
            int(raw.get("shard_size_mb", 512)) * 2**20,
            args.revision,
            checkpoint_seconds,
            checkpoint_records,
            retry_policy,
            args.max_rss_gib,
        )
        manifest["updated_at_unix"] = time.time()
        atomic_write_json(manifest_path, manifest)
    print(f"Stack-Edu corpus is under {output}")


def _cli_entrypoint() -> None:
    try:
        main()
    except MemoryPressureError as exc:
        print(str(exc), file=sys.stderr, flush=True)
        raise SystemExit(75) from exc
    sys.stdout.flush()
    sys.stderr.flush()
    if os.environ.get("ASTERLM_MATERIALIZER_HARD_EXIT", "1") != "0":
        os._exit(0)


if __name__ == "__main__":
    _cli_entrypoint()
