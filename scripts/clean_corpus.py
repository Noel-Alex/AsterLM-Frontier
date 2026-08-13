#!/usr/bin/env python
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import sqlite3
import time
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from tqdm import tqdm

from asterlm.data.quality import (
    benchmark_ngrams,
    contamination_fraction,
    exact_digest,
    hamming64,
    quality_decision,
    simhash64,
)


def iter_records(root: Path) -> Iterator[tuple[dict[str, Any], Path]]:
    for path in record_paths(root):
        yield from iter_path_records(path)


def record_paths(root: Path) -> list[Path]:
    candidates = root.rglob("*") if root.is_dir() else [root]
    return sorted(path for path in candidates if path.is_file() and is_record_path(path))


def is_record_path(path: Path) -> bool:
    lower = path.name.lower()
    return lower.endswith((".jsonl.zst", ".jsonl.gz", ".jsonl", ".txt", ".md"))


def iter_path_records(path: Path) -> Iterator[tuple[dict[str, Any], Path]]:
    lower = path.name.lower()
    if lower.endswith(".jsonl.zst"):
        import zstandard as zstd

        raw = path.open("rb")
        reader = zstd.ZstdDecompressor().stream_reader(raw)
        handle = io.TextIOWrapper(reader, encoding="utf-8")
    elif lower.endswith(".jsonl.gz"):
        handle = gzip.open(path, "rt", encoding="utf-8")  # noqa: SIM115
    elif lower.endswith(".jsonl"):
        handle = path.open("r", encoding="utf-8")
    elif lower.endswith((".txt", ".md")):
        yield {"text": path.read_text(encoding="utf-8", errors="replace")}, path
        return
    else:
        return
    with handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                yield record, path


def nested_get(record: dict[str, Any], field: str) -> Any:
    value: Any = record
    for part in field.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def is_validation_digest(digest: bytes, fraction: float) -> bool:
    return int.from_bytes(digest[:8], "big") / float(2**64) < fraction


class ShardWriter:
    def __init__(self, root: Path, shard_mb: int) -> None:
        self.root = root
        self.limit = shard_mb * 2**20
        self.root.mkdir(parents=True, exist_ok=True)
        existing = []
        for path in self.root.glob("clean-*.jsonl.zst"):
            try:
                existing.append(int(path.name.split("-")[1].split(".")[0]))
            except (IndexError, ValueError):
                continue
        self.index = (max(existing) + 1) if existing else 0
        self.bytes = 0
        self.raw = self.stream = self.text = None

    def _open(self) -> None:
        import zstandard as zstd

        self.root.mkdir(parents=True, exist_ok=True)
        partial = self.root / f"clean-{self.index:05d}.jsonl.zst.partial"
        if partial.exists():
            partial.unlink()
        self.raw = partial.open("wb")
        self.stream = zstd.ZstdCompressor(level=6, threads=0).stream_writer(self.raw)
        self.text = io.TextIOWrapper(self.stream, encoding="utf-8", write_through=True)
        self.bytes = 0

    def write(self, record: dict[str, Any]) -> Path | None:
        if self.text is None:
            self._open()
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        self.text.write(line)
        self.bytes += len(line.encode("utf-8"))
        if self.bytes >= self.limit:
            return self.close()
        return None

    def close(self) -> Path | None:
        if self.text is None:
            return None
        partial = Path(self.raw.name)
        self.text.flush()
        self.text.detach()
        self.stream.close()
        self.raw.close()
        completed = Path(str(partial).removesuffix(".partial"))
        os.replace(partial, completed)
        self.index += 1
        self.raw = self.stream = self.text = None
        return completed


def init_db(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("CREATE TABLE IF NOT EXISTS exact(hash BLOB PRIMARY KEY)")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS simhash(id INTEGER PRIMARY KEY, value INTEGER NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS bands(band INTEGER, bucket INTEGER, sim_id INTEGER, "
        "PRIMARY KEY(band,bucket,sim_id))"
    )
    connection.execute("CREATE INDEX IF NOT EXISTS idx_bands ON bands(band,bucket)")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS clean_outputs("
        "path TEXT PRIMARY KEY, job_key TEXT NOT NULL, size INTEGER NOT NULL, "
        "mtime_ns INTEGER NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS clean_progress("
        "job_key TEXT PRIMARY KEY, input_manifest_sha256 TEXT NOT NULL, "
        "last_completed_path TEXT)"
    )
    return connection


def signed64(value: int) -> int:
    return value - 2**64 if value >= 2**63 else value


def unsigned64(value: int) -> int:
    return value + 2**64 if value < 0 else value


def is_near_duplicate(connection: sqlite3.Connection, value: int, distance: int) -> bool:
    candidates: set[int] = set()
    for band in range(4):
        bucket = (value >> (16 * band)) & 0xFFFF
        rows = connection.execute(
            "SELECT sim_id FROM bands WHERE band=? AND bucket=?", (band, bucket)
        )
        candidates.update(row[0] for row in rows)
    if not candidates:
        return False
    placeholders = ",".join("?" for _ in candidates)
    rows = connection.execute(
        f"SELECT value FROM simhash WHERE id IN ({placeholders})", tuple(candidates)
    )
    return any(hamming64(value, unsigned64(row[0])) <= distance for row in rows)


def insert_hashes(connection: sqlite3.Connection, digest: bytes, value: int) -> None:
    connection.execute("INSERT INTO exact(hash) VALUES (?)", (digest,))
    cursor = connection.execute("INSERT INTO simhash(value) VALUES (?)", (signed64(value),))
    sim_id = int(cursor.lastrowid)
    connection.executemany(
        "INSERT INTO bands(band,bucket,sim_id) VALUES (?,?,?)",
        [(band, (value >> (16 * band)) & 0xFFFF, sim_id) for band in range(4)],
    )


def source_manifest(root: Path, paths: list[Path]) -> str:
    base = root if root.is_dir() else root.parent
    rows = []
    for path in paths:
        stat = path.stat()
        rows.append(
            {
                "path": path.relative_to(base).as_posix(),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def cleaning_job_key(input_root: Path, output: Path, validation_output: Path | None) -> str:
    fields = [str(input_root.resolve()), str(output.resolve())]
    fields.append(str(validation_output.resolve()) if validation_output else "")
    return hashlib.sha256("\0".join(fields).encode("utf-8")).hexdigest()


def load_progress(
    connection: sqlite3.Connection,
    *,
    job_key: str,
    manifest_sha256: str,
) -> str | None:
    row = connection.execute(
        "SELECT input_manifest_sha256,last_completed_path FROM clean_progress WHERE job_key=?",
        (job_key,),
    ).fetchone()
    if row is None:
        connection.execute(
            "INSERT INTO clean_progress(job_key,input_manifest_sha256,last_completed_path) "
            "VALUES (?,?,NULL)",
            (job_key, manifest_sha256),
        )
        connection.commit()
        return None
    if row[0] != manifest_sha256:
        raise RuntimeError(
            "Input shard manifest changed after cleaning began; reset the clean output "
            "instead of blending two source snapshots"
        )
    return row[1]


def register_outputs(
    connection: sqlite3.Connection, *, job_key: str, paths: list[Path]
) -> None:
    for path in paths:
        stat = path.stat()
        connection.execute(
            "INSERT OR REPLACE INTO clean_outputs(path,job_key,size,mtime_ns) VALUES (?,?,?,?)",
            (str(path.resolve()), job_key, stat.st_size, stat.st_mtime_ns),
        )


def recover_finalized_outputs(
    connection: sqlite3.Connection,
    *,
    job_key: str,
    roots: list[Path],
    text_field: str,
) -> int:
    registered = {
        Path(path): (int(size), int(mtime_ns))
        for path, size, mtime_ns in connection.execute(
            "SELECT path,size,mtime_ns FROM clean_outputs WHERE job_key=?", (job_key,)
        )
    }
    for path, (size, mtime_ns) in registered.items():
        if not path.is_file():
            raise RuntimeError(
                f"Committed clean shard is missing: {path}. Restore it or reset the clean output."
            )
        stat = path.stat()
        if stat.st_size != size or stat.st_mtime_ns != mtime_ns:
            raise RuntimeError(f"Committed clean shard changed after registration: {path}")

    finalized = sorted(path.resolve() for root in roots for path in root.glob("clean-*.jsonl.zst"))
    recovered = 0
    for path in finalized:
        if path in registered:
            continue
        for record, _ in iter_path_records(path):
            text = nested_get(record, text_field)
            if text is None:
                raise RuntimeError(f"Finalized clean shard lacks {text_field!r}: {path}")
            normalized = str(text)
            digest = exact_digest(normalized)
            if connection.execute("SELECT 1 FROM exact WHERE hash=?", (digest,)).fetchone():
                continue
            insert_hashes(connection, digest, simhash64(normalized))
        register_outputs(connection, job_key=job_key, paths=[path])
        recovered += 1
    connection.commit()
    return recovered


def checkpoint_cleaning(
    connection: sqlite3.Connection,
    *,
    job_key: str,
    manifest_sha256: str,
    last_completed_path: str | None,
    writers: list[ShardWriter],
    completed_paths: list[Path],
) -> None:
    for writer in writers:
        completed = writer.close()
        if completed is not None:
            completed_paths.append(completed)
    register_outputs(connection, job_key=job_key, paths=completed_paths)
    completed_paths.clear()
    connection.execute(
        "UPDATE clean_progress SET input_manifest_sha256=?,last_completed_path=? WHERE job_key=?",
        (manifest_sha256, last_completed_path, job_key),
    )
    connection.commit()


def _flatten_strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _flatten_strings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _flatten_strings(child)


def load_benchmark_texts(paths: list[str]) -> list[str]:
    texts: list[str] = []
    for raw in paths:
        path = Path(raw)
        for record, _ in iter_records(path):
            # Hash questions, answer choices, tests, and reference code. Metadata fields
            # add little because short n-gram windows naturally disappear.
            joined = "\n".join(_flatten_strings(record))
            if joined:
                texts.append(joined)
    return texts


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Quality filter, deduplicate, redact, and decontaminate a local corpus"
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--dedup-db",
        default=None,
        help="Optional shared SQLite path for exact/near deduplication across sources",
    )
    parser.add_argument(
        "--validation-output",
        default=None,
        help="Optional disjoint deterministic validation output directory",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.0,
        help="Fraction of accepted unique documents routed only to validation",
    )
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--source-id", default=None)
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--max-chars", type=int, default=500000)
    parser.add_argument("--pii-mode", choices=["redact", "drop", "keep"], default="redact")
    parser.add_argument("--near-distance", type=int, default=3)
    parser.add_argument("--benchmark", action="append", default=[])
    parser.add_argument("--max-contamination", type=float, default=0.0)
    parser.add_argument("--shard-mb", type=int, default=512)
    parser.add_argument(
        "--commit-every",
        type=int,
        default=100000,
        help="Finalize shards and atomically checkpoint dedup/source progress every N input records",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if not 0.0 <= args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be in [0, 1)")
    if args.validation_fraction > 0 and not args.validation_output:
        raise ValueError("--validation-output is required when --validation-fraction is positive")
    if args.commit_every <= 0:
        raise ValueError("--commit-every must be positive")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    input_root = Path(args.input)
    input_paths = record_paths(input_root)
    if not input_paths:
        raise FileNotFoundError(f"No supported records found under {input_root}")
    dedup_db = Path(args.dedup_db) if args.dedup_db else output / "dedup.sqlite"
    dedup_db.parent.mkdir(parents=True, exist_ok=True)
    connection = init_db(dedup_db)
    writer = ShardWriter(output, args.shard_mb)
    validation_output = Path(args.validation_output) if args.validation_output else None
    validation_writer = (
        ShardWriter(validation_output, args.shard_mb) if validation_output else None
    )
    writers = [writer, *([validation_writer] if validation_writer is not None else [])]
    manifest_sha256 = source_manifest(input_root, input_paths)
    job_key = cleaning_job_key(input_root, output, validation_output)
    last_completed_path = load_progress(
        connection, job_key=job_key, manifest_sha256=manifest_sha256
    )
    path_base = input_root if input_root.is_dir() else input_root.parent
    path_keys = [path.relative_to(path_base).as_posix() for path in input_paths]
    if last_completed_path is not None and last_completed_path not in path_keys:
        raise RuntimeError(
            f"Saved clean-corpus cursor no longer exists in the input: {last_completed_path}"
        )
    start_index = path_keys.index(last_completed_path) + 1 if last_completed_path else 0
    recovered_shards = recover_finalized_outputs(
        connection,
        job_key=job_key,
        roots=[output, *([validation_output] if validation_output else [])],
        text_field=args.text_field,
    )
    benchmark_hashes = benchmark_ngrams(load_benchmark_texts(args.benchmark))
    reasons: Counter[str] = Counter()
    kept_chars = 0
    seen = 0
    started = time.time()
    progress = tqdm(desc="clean", unit="doc")
    completed_paths: list[Path] = []
    limit_reached = False
    try:
        for source_path, source_key in zip(
            input_paths[start_index:], path_keys[start_index:], strict=True
        ):
            for record, _ in iter_path_records(source_path):
                if args.limit is not None and seen >= args.limit:
                    limit_reached = True
                    break
                seen += 1
                progress.update(1)
                if seen % args.commit_every == 0:
                    checkpoint_cleaning(
                        connection,
                        job_key=job_key,
                        manifest_sha256=manifest_sha256,
                        last_completed_path=last_completed_path,
                        writers=writers,
                        completed_paths=completed_paths,
                    )
                raw_text = nested_get(record, args.text_field)
                if raw_text is None:
                    reasons["missing_text"] += 1
                    continue
                decision = quality_decision(
                    str(raw_text),
                    min_chars=args.min_chars,
                    max_chars=args.max_chars,
                    pii_mode=args.pii_mode,
                )
                if not decision.keep:
                    reasons[decision.reason] += 1
                    continue
                text = decision.normalized_text
                digest = exact_digest(text)
                if connection.execute("SELECT 1 FROM exact WHERE hash=?", (digest,)).fetchone():
                    reasons["exact_duplicate"] += 1
                    continue
                sim = simhash64(text)
                if is_near_duplicate(connection, sim, args.near_distance):
                    reasons["near_duplicate"] += 1
                    continue
                contamination = contamination_fraction(text, benchmark_hashes)
                if contamination > args.max_contamination:
                    reasons["benchmark_contamination"] += 1
                    continue
                insert_hashes(connection, digest, sim)
                clean = dict(record)
                clean[args.text_field] = text
                clean["_source_file"] = str(source_path)
                if args.source_id:
                    clean["_source_id"] = args.source_id
                clean["_quality"] = decision.metrics
                clean["_contamination_fraction"] = contamination
                if validation_writer is not None and is_validation_digest(
                    digest, args.validation_fraction
                ):
                    completed = validation_writer.write(clean)
                    if completed is not None:
                        completed_paths.append(completed)
                    reasons["validation_kept"] += 1
                else:
                    completed = writer.write(clean)
                    if completed is not None:
                        completed_paths.append(completed)
                    reasons["train_kept"] += 1
                reasons["kept"] += 1
                kept_chars += len(text)
            if limit_reached:
                break
            last_completed_path = source_key
    finally:
        checkpoint_cleaning(
            connection,
            job_key=job_key,
            manifest_sha256=manifest_sha256,
            last_completed_path=last_completed_path,
            writers=writers,
            completed_paths=completed_paths,
        )
        connection.close()
        progress.close()

    report = {
        "input": args.input,
        "output": args.output,
        "seen": seen,
        "counts": dict(reasons),
        "kept_chars": kept_chars,
        "estimated_tokens": round(kept_chars / 4),
        "benchmark_hashes": len(benchmark_hashes),
        "elapsed_seconds": time.time() - started,
        "arguments": vars(args),
        "dedup_db": str(dedup_db),
        "source_manifest_sha256": manifest_sha256,
        "last_completed_path": last_completed_path,
        "recovered_finalized_shards": recovered_shards,
        "resume_start_file_index": start_index,
    }
    (output / "cleaning_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
