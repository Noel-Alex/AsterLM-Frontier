from __future__ import annotations

import hashlib
import json
import os
import pickle
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator


@dataclass(slots=True)
class RetryPolicy:
    max_retries: int = 50
    base_seconds: float = 5.0
    max_seconds: float = 300.0
    jitter: float = 0.20

    def delay(self, attempt: int) -> float:
        raw = min(self.max_seconds, self.base_seconds * (2 ** max(0, attempt - 1)))
        if self.jitter <= 0:
            return raw
        return max(0.0, raw * random.uniform(1.0 - self.jitter, 1.0 + self.jitter))

    def permits(self, failures: int) -> bool:
        return self.max_retries == 0 or failures <= self.max_retries


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def sha256_file(path: Path, chunk_bytes: int = 8 * 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def is_retryable_exception(exc: BaseException) -> bool:
    """Conservative network/transient classifier.

    Unknown exceptions are not retried automatically because schema, auth, and
    programming errors should fail loudly rather than loop for hours.
    """

    retryable_types: tuple[type[BaseException], ...] = (
        TimeoutError,
        ConnectionError,
        EOFError,
        OSError,
    )
    if isinstance(exc, retryable_types):
        text = str(exc).lower()
        permanent_markers = (
            "permission denied",
            "no space left",
            "read-only file system",
            "not supported",
            "invalid config",
            "invalid split",
            "doesn't exist",
            "not found",
            "401",
            "403",
        )
        return not any(marker in text for marker in permanent_markers)

    text = f"{type(exc).__name__}: {exc}".lower()
    transient_markers = (
        "timeout",
        "timed out",
        "connection reset",
        "connection aborted",
        "connection refused",
        "remote disconnected",
        "temporary failure",
        "temporarily unavailable",
        "server disconnected",
        "could not connect",
        "endpoint connection",
        "incomplete read",
        "chunkedencodingerror",
        "protocolerror",
        "502 bad gateway",
        "503 service unavailable",
        "504 gateway timeout",
        "429 too many requests",
        "rate limit",
    )
    return any(marker in text for marker in transient_markers)


def dataset_cursor(dataset: Any) -> Any | None:
    method = getattr(dataset, "state_dict", None)
    return method() if callable(method) else None


def restore_dataset_cursor(dataset: Any, cursor: Any | None, fallback_skip: int) -> Any:
    if cursor is not None:
        method = getattr(dataset, "load_state_dict", None)
        if callable(method):
            method(cursor)
            return dataset
    if fallback_skip:
        method = getattr(dataset, "skip", None)
        if callable(method):
            return method(fallback_skip)
    return dataset


def newest_committed_cursor(root: Path, state: dict[str, Any]) -> Any | None:
    cursor_name = state.get("cursor_file")
    if not cursor_name:
        return None
    cursor_path = root / str(cursor_name)
    if not cursor_path.exists():
        raise RuntimeError(
            f"Checkpoint state references missing cursor file {cursor_path}. "
            "Restore the file or remove state.json and the corresponding output shards."
        )
    return load_pickle(cursor_path)


def clean_uncommitted_outputs(
    root: Path, prefix: str, next_shard_index: int, checkpoint_id: int = 0
) -> list[str]:
    """Remove partial/orphan artifacts newer than the atomic state pointer.

    A hard interruption can happen after a data shard was renamed but before
    state.json was advanced. Such an orphan is intentionally discarded to avoid
    duplication. The loss is bounded by the checkpoint interval.
    """

    removed: list[str] = []
    root.mkdir(parents=True, exist_ok=True)
    for partial in root.glob(f"{prefix}-*.partial"):
        partial.unlink(missing_ok=True)
        removed.append(str(partial))
    for temp in root.glob("cursor-*.pkl.tmp"):
        temp.unlink(missing_ok=True)
        removed.append(str(temp))
    for path in root.glob(f"{prefix}-*.jsonl.zst"):
        try:
            index = int(path.name.removeprefix(f"{prefix}-").split(".", 1)[0])
        except (ValueError, IndexError):
            continue
        if index >= next_shard_index:
            path.unlink(missing_ok=True)
            removed.append(str(path))
    for cursor in root.glob("cursor-*.pkl"):
        try:
            index = int(cursor.stem.split("-", 1)[1])
        except (ValueError, IndexError):
            continue
        if index > checkpoint_id:
            cursor.unlink(missing_ok=True)
            removed.append(str(cursor))
    return removed


class ZstdCheckpointWriter:
    """One-open-shard writer with zstd frame checksums.

    A shard is only considered durable after the caller atomically advances
    state.json. The caller may safely discard any orphan shard newer than that
    pointer after an unclean shutdown.
    """

    def __init__(self, root: Path, prefix: str, shard_bytes: int, index: int) -> None:
        self.root = root
        self.prefix = prefix
        self.shard_bytes = shard_bytes
        self.index = index
        self.raw: Any | None = None
        self.stream: Any | None = None
        self.text: Any | None = None
        self.uncompressed_bytes = 0
        self.records = 0
        self.opened_at = 0.0

    @property
    def is_open(self) -> bool:
        return self.text is not None

    def _open(self) -> None:
        import io
        import zstandard as zstd

        self.root.mkdir(parents=True, exist_ok=True)
        partial = self.root / f"{self.prefix}-{self.index:05d}.jsonl.zst.partial"
        self.raw = partial.open("wb")
        level = int(os.getenv("ASTERLM_ZSTD_LEVEL", "6"))
        threads = int(os.getenv("ASTERLM_ZSTD_THREADS", "0"))
        compressor = zstd.ZstdCompressor(level=level, threads=threads, write_checksum=True)
        self.stream = compressor.stream_writer(self.raw, closefd=False)
        self.text = io.TextIOWrapper(self.stream, encoding="utf-8", write_through=True)
        self.uncompressed_bytes = 0
        self.records = 0
        self.opened_at = time.monotonic()

    def write(self, record: dict[str, Any]) -> None:
        if self.text is None:
            self._open()
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        self.text.write(line)
        self.uncompressed_bytes += len(line.encode("utf-8"))
        self.records += 1

    def should_checkpoint(self, checkpoint_seconds: float) -> bool:
        if not self.is_open:
            return False
        return self.uncompressed_bytes >= self.shard_bytes or (
            checkpoint_seconds > 0 and time.monotonic() - self.opened_at >= checkpoint_seconds
        )

    def finalize(self) -> dict[str, Any] | None:
        if self.text is None:
            return None
        partial = Path(self.raw.name)
        self.text.flush()
        self.text.detach()
        self.stream.close()
        self.raw.flush()
        os.fsync(self.raw.fileno())
        self.raw.close()
        final = Path(str(partial).removesuffix(".partial"))
        os.replace(partial, final)
        info = {
            "path": str(final),
            "name": final.name,
            "index": self.index,
            "records": self.records,
            "uncompressed_bytes": self.uncompressed_bytes,
            "compressed_bytes": final.stat().st_size,
        }
        self.index += 1
        self.raw = self.stream = self.text = None
        self.uncompressed_bytes = 0
        self.records = 0
        self.opened_at = 0.0
        return info

    def discard_partial(self) -> None:
        if self.text is not None:
            partial = Path(self.raw.name)
            try:
                self.text.detach()
            except Exception:
                pass
            try:
                self.stream.close()
            except Exception:
                pass
            try:
                self.raw.close()
            except Exception:
                pass
            partial.unlink(missing_ok=True)
        self.raw = self.stream = self.text = None


def retry_loop(
    operation: Callable[[int], Any],
    policy: RetryPolicy,
    *,
    label: str,
    on_error: Callable[[BaseException, int, float], None] | None = None,
) -> Any:
    failures = 0
    while True:
        try:
            return operation(failures)
        except KeyboardInterrupt:
            raise
        except BaseException as exc:
            failures += 1
            if not is_retryable_exception(exc) or not policy.permits(failures):
                raise
            delay = policy.delay(failures)
            if on_error:
                on_error(exc, failures, delay)
            else:
                print(f"{label}: transient failure {failures}: {type(exc).__name__}: {exc}; retrying in {delay:.1f}s")
            time.sleep(delay)
