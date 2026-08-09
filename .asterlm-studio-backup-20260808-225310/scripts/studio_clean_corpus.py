#!/usr/bin/env python
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from asterlm.data.quality import (
    benchmark_ngrams,
    contamination_fraction,
    exact_digest,
    quality_decision,
    simhash64,
)
from clean_corpus import (
    ShardWriter,
    init_db,
    insert_hashes,
    is_near_duplicate,
    is_validation_digest,
    iter_records,
    load_benchmark_texts,
    nested_get,
)

ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def data_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    suffixes = (".jsonl.zst", ".jsonl.gz", ".jsonl", ".txt", ".md")
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name.lower().endswith(suffixes)
    )


def plan_signature(plan: dict[str, Any]) -> str:
    relevant = {
        "validation_fraction": plan.get("validation_fraction"),
        "pii_mode": plan.get("pii_mode"),
        "near_distance": plan.get("near_distance"),
        "max_contamination": plan.get("max_contamination", 0.0),
        "benchmarks": plan.get("benchmarks"),
        "sources": [
            {
                "id": item.get("id"),
                "raw_path": item.get("raw_path"),
                "text_field": item.get("text_field", "text"),
            }
            for item in plan.get("sources", [])
        ],
    }
    return hashlib.sha256(
        json.dumps(relevant, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def reconcile_existing_outputs(
    output: Path,
    connection: sqlite3.Connection,
    source_ids: list[str],
) -> int:
    """Reinsert hashes from durable clean shards after an unclean machine crash.

    A marker is left while Studio is processing. Graceful Ctrl+C closes writers
    and commits SQLite before removing it. If power is lost between file and DB
    durability, the next run scans only the already-cleaned derivative and makes
    the DB agree with those durable outputs before resuming raw input.
    """

    inserted = 0
    roots = [output / sid for sid in source_ids]
    roots += [output / "validation" / sid for sid in source_ids]
    for root in roots:
        if not root.exists():
            continue
        for record, _ in iter_records(root):
            value = record.get("text")
            if not isinstance(value, str) or not value:
                continue
            digest = exact_digest(value)
            if connection.execute(
                "SELECT 1 FROM exact WHERE hash=?",
                (digest,),
            ).fetchone():
                continue
            sim = simhash64(value)
            insert_hashes(connection, digest, sim)
            inserted += 1
            if inserted % 10_000 == 0:
                connection.commit()
    connection.commit()
    return inserted


def checkpoint(
    *,
    writers: dict[str, ShardWriter],
    validation_writers: dict[str, ShardWriter],
    connection: sqlite3.Connection,
    state_path: Path,
    state: dict[str, Any],
) -> None:
    # Finalize compressed frames before advancing the input cursor.
    for writer in writers.values():
        writer.close()
    for writer in validation_writers.values():
        writer.close()
    connection.commit()
    atomic_json(state_path, state)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Globally deduplicate/decontaminate an arbitrary multi-source Aster Studio corpus"
    )
    parser.add_argument("--plan", required=True)
    parser.add_argument("--checkpoint-records", type=int, default=100_000)
    parser.add_argument("--shard-mb", type=int, default=512)
    parser.add_argument("--allow-no-benchmarks", action="store_true")
    args = parser.parse_args()

    plan_path = Path(args.plan)
    if not plan_path.is_absolute():
        plan_path = ROOT / plan_path
    raw = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    plan = raw.get("plan", raw)
    output = ROOT / str(plan["output"])
    sources = list(plan.get("sources", []))
    if not sources:
        raise SystemExit("Cleaning plan has no sources")

    validation_fraction = float(plan.get("validation_fraction", 0.005))
    pii_mode = str(plan.get("pii_mode", "redact"))
    near_distance = int(plan.get("near_distance", 3))
    max_contamination = float(plan.get("max_contamination", 0.0))
    benchmarks = ROOT / str(plan["benchmarks"])
    if not benchmarks.exists() and not args.allow_no_benchmarks:
        raise FileNotFoundError(
            f"Benchmark/decontamination directory is missing: {benchmarks}"
        )

    signature = plan_signature(plan)
    output.mkdir(parents=True, exist_ok=True)
    signature_path = output / "_studio_clean_signature.json"
    prior_signature = None
    if signature_path.exists():
        prior_signature = json.loads(signature_path.read_text(encoding="utf-8")).get(
            "signature"
        )
    if prior_signature and prior_signature != signature:
        raise RuntimeError(
            "The existing clean derivative was created with a different source/"
            "decontamination transformation. Use the Studio reset-clean option "
            "instead of mixing two transformations into one output."
        )
    if not signature_path.exists():
        atomic_json(
            signature_path,
            {
                "signature": signature,
                "plan": str(plan_path),
                "created_at_unix": time.time(),
            },
        )

    state_path = output / "_studio_clean_state.json"
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.exists()
        else {
            "version": 1,
            "signature": signature,
            "source_index": 0,
            "file_index": 0,
            "record_index": 0,
            "seen": 0,
            "kept": 0,
            "kept_chars": 0,
            "started_at_unix": time.time(),
            "complete": False,
            "per_source": {},
        }
    )
    if state.get("signature") != signature:
        raise RuntimeError("Cleaning state signature does not match the requested plan")
    if state.get("complete"):
        print(
            f"Studio clean derivative is already complete: {output}",
            flush=True,
        )
        return

    db_path = output / "_studio_global_dedup.sqlite"
    connection = init_db(db_path)
    # Prefer durability for a multi-hour clean campaign.
    connection.execute("PRAGMA synchronous=FULL")

    marker = output / "_studio_clean_recovery_needed"
    source_ids = [str(item["id"]) for item in sources]
    if marker.exists():
        print(
            "Detected an unclean prior shutdown; reconciling durable clean shards "
            "with the global dedup index...",
            flush=True,
        )
        inserted = reconcile_existing_outputs(output, connection, source_ids)
        print(f"Reconciled {inserted:,} durable document hash(es).", flush=True)
    marker.write_text(f"pid={os.getpid()} started={time.time()}\n", encoding="utf-8")

    benchmark_hashes = (
        benchmark_ngrams(load_benchmark_texts([str(benchmarks)]))
        if benchmarks.exists()
        else set()
    )
    if benchmark_hashes:
        print(f"Loaded {len(benchmark_hashes):,} benchmark n-grams for decontamination.")
    else:
        print("WARNING: cleaning without benchmark decontamination was explicitly requested.")

    writers: dict[str, ShardWriter] = {}
    validation_writers: dict[str, ShardWriter] = {}
    since_checkpoint = 0
    started = time.time()
    progress = tqdm(
        initial=int(state.get("seen", 0)),
        desc="global clean",
        unit="doc",
        dynamic_ncols=True,
    )

    try:
        for source_index in range(int(state["source_index"]), len(sources)):
            item = sources[source_index]
            sid = str(item["id"])
            text_field = str(item.get("text_field", "text"))
            raw_root = ROOT / str(item["raw_path"])
            files = data_files(raw_root)
            if not files:
                if bool(item.get("optional", False)):
                    print(f"Skipping optional empty source {sid}: {raw_root}")
                    state.update(
                        {
                            "source_index": source_index + 1,
                            "file_index": 0,
                            "record_index": 0,
                        }
                    )
                    checkpoint(
                        writers=writers,
                        validation_writers=validation_writers,
                        connection=connection,
                        state_path=state_path,
                        state=state,
                    )
                    continue
                raise RuntimeError(f"No supported raw data shards for {sid}: {raw_root}")

            per_source = state["per_source"].setdefault(
                sid,
                {
                    "seen": 0,
                    "kept": 0,
                    "kept_chars": 0,
                    "counts": {},
                },
            )
            counts = collections.Counter(per_source.get("counts", {}))
            start_file = int(state["file_index"]) if source_index == int(state["source_index"]) else 0

            for file_index in range(start_file, len(files)):
                path = files[file_index]
                start_record = (
                    int(state["record_index"])
                    if source_index == int(state["source_index"])
                    and file_index == int(state["file_index"])
                    else 0
                )
                for record_index, (record, source_path) in enumerate(
                    iter_records(path)
                ):
                    if record_index < start_record:
                        continue

                    state["seen"] = int(state.get("seen", 0)) + 1
                    per_source["seen"] = int(per_source.get("seen", 0)) + 1
                    progress.update(1)

                    raw_text = nested_get(record, text_field)
                    if raw_text is None:
                        counts["missing_text"] += 1
                    else:
                        decision = quality_decision(
                            str(raw_text),
                            min_chars=int(plan.get("min_chars", 200)),
                            max_chars=int(plan.get("max_chars", 500000)),
                            pii_mode=pii_mode,
                        )
                        if not decision.keep:
                            counts[decision.reason] += 1
                        else:
                            text = decision.normalized_text
                            digest = exact_digest(text)
                            if connection.execute(
                                "SELECT 1 FROM exact WHERE hash=?",
                                (digest,),
                            ).fetchone():
                                counts["exact_duplicate"] += 1
                            else:
                                sim = simhash64(text)
                                if is_near_duplicate(connection, sim, near_distance):
                                    counts["near_duplicate"] += 1
                                else:
                                    contamination = contamination_fraction(
                                        text,
                                        benchmark_hashes,
                                    )
                                    if contamination > max_contamination:
                                        counts["benchmark_contamination"] += 1
                                    else:
                                        insert_hashes(connection, digest, sim)
                                        clean = dict(record)
                                        clean[text_field] = text
                                        clean["_source_file"] = str(source_path)
                                        clean["_source_id"] = sid
                                        clean["_quality"] = decision.metrics
                                        clean["_contamination_fraction"] = contamination
                                        if (
                                            validation_fraction > 0
                                            and is_validation_digest(
                                                digest,
                                                validation_fraction,
                                            )
                                        ):
                                            writer = validation_writers.setdefault(
                                                sid,
                                                ShardWriter(
                                                    output / "validation" / sid,
                                                    args.shard_mb,
                                                ),
                                            )
                                            writer.write(clean)
                                            counts["validation_kept"] += 1
                                        else:
                                            writer = writers.setdefault(
                                                sid,
                                                ShardWriter(
                                                    output / sid,
                                                    args.shard_mb,
                                                ),
                                            )
                                            writer.write(clean)
                                            counts["train_kept"] += 1
                                        counts["kept"] += 1
                                        chars = len(text)
                                        state["kept"] = int(state.get("kept", 0)) + 1
                                        state["kept_chars"] = int(
                                            state.get("kept_chars", 0)
                                        ) + chars
                                        per_source["kept"] = int(
                                            per_source.get("kept", 0)
                                        ) + 1
                                        per_source["kept_chars"] = int(
                                            per_source.get("kept_chars", 0)
                                        ) + chars

                    per_source["counts"] = dict(counts)
                    # Input cursor always names the next record.
                    state.update(
                        {
                            "source_index": source_index,
                            "file_index": file_index,
                            "record_index": record_index + 1,
                        }
                    )
                    since_checkpoint += 1
                    if (
                        args.checkpoint_records > 0
                        and since_checkpoint >= args.checkpoint_records
                    ):
                        checkpoint(
                            writers=writers,
                            validation_writers=validation_writers,
                            connection=connection,
                            state_path=state_path,
                            state=state,
                        )
                        since_checkpoint = 0

                state.update(
                    {
                        "source_index": source_index,
                        "file_index": file_index + 1,
                        "record_index": 0,
                    }
                )
                checkpoint(
                    writers=writers,
                    validation_writers=validation_writers,
                    connection=connection,
                    state_path=state_path,
                    state=state,
                )
                since_checkpoint = 0

            state.update(
                {
                    "source_index": source_index + 1,
                    "file_index": 0,
                    "record_index": 0,
                }
            )
            checkpoint(
                writers=writers,
                validation_writers=validation_writers,
                connection=connection,
                state_path=state_path,
                state=state,
            )
            since_checkpoint = 0

        state["complete"] = True
        state["finished_at_unix"] = time.time()
        checkpoint(
            writers=writers,
            validation_writers=validation_writers,
            connection=connection,
            state_path=state_path,
            state=state,
        )
        marker.unlink(missing_ok=True)

    except KeyboardInterrupt:
        print("\nCleaning stop requested; committing current clean shards and cursor...")
        checkpoint(
            writers=writers,
            validation_writers=validation_writers,
            connection=connection,
            state_path=state_path,
            state=state,
        )
        marker.unlink(missing_ok=True)
        raise
    finally:
        for writer in writers.values():
            writer.close()
        for writer in validation_writers.values():
            writer.close()
        connection.commit()
        connection.close()
        progress.close()

    report = {
        "version": 1,
        "plan": str(plan_path),
        "signature": signature,
        "output": str(output),
        "seen": int(state.get("seen", 0)),
        "kept": int(state.get("kept", 0)),
        "kept_chars": int(state.get("kept_chars", 0)),
        "estimated_tokens": round(int(state.get("kept_chars", 0)) / 4),
        "benchmark_hashes": len(benchmark_hashes),
        "elapsed_seconds_this_process": time.time() - started,
        "per_source": state.get("per_source", {}),
    }
    atomic_json(output / "studio_global_cleaning_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
