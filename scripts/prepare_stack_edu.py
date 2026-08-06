#!/usr/bin/env python
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from typing import Any

from tqdm import tqdm


_thread_local = threading.local()


def _s3_client():
    client = getattr(_thread_local, "s3", None)
    if client is None:
        try:
            import boto3
            from botocore import UNSIGNED
            from botocore.config import Config
        except ImportError as exc:
            raise RuntimeError("Install boto3: python -m pip install boto3") from exc
        client = boto3.client(
            "s3",
            config=Config(signature_version=UNSIGNED, retries={"max_attempts": 8, "mode": "adaptive"}),
        )
        _thread_local.s3 = client
    return client


def fetch_record(record: dict[str, Any]) -> dict[str, Any] | None:
    # The dataset exposes license_type explicitly. Default to fail-closed.
    if record.get("license_type") != "permissive":
        return None
    blob_id = str(record["blob_id"])
    try:
        obj = _s3_client().get_object(Bucket="softwareheritage", Key=f"content/{blob_id}")
        with gzip.GzipFile(fileobj=obj["Body"]) as fin:
            raw = fin.read()
        encoding = str(record.get("src_encoding") or "utf-8")
        try:
            text = raw.decode(encoding, errors="replace")
        except LookupError:
            text = raw.decode("utf-8", errors="replace")
    except Exception:
        return None
    if not text.strip():
        return None
    return {
        "text": text,
        "blob_id": blob_id,
        "repo_name": record.get("repo_name"),
        "path": record.get("path"),
        "license_type": record.get("license_type"),
        "detected_licenses": record.get("detected_licenses"),
        "score": record.get("score"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize permissively licensed Stack-Edu code from Software Heritage S3"
    )
    parser.add_argument("--output", default="data/stack_edu_python.jsonl")
    parser.add_argument("--language", default="Python")
    parser.add_argument("--documents", type=int, default=1_000_000)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--in-flight", type=int, default=256)
    parser.add_argument("--revision", default=None, help="Optional Hugging Face dataset revision")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install project dependencies first: pip install -e .") from exc

    kwargs: dict[str, Any] = dict(
        path="HuggingFaceTB/stack-edu",
        name=args.language,
        split="train",
        streaming=True,
    )
    if args.revision:
        kwargs["revision"] = args.revision
    dataset = load_dataset(**kwargs)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    submitted = 0
    pending = set()
    with ThreadPoolExecutor(max_workers=args.workers) as pool, output.open("w", encoding="utf-8") as handle:
        progress = tqdm(total=args.documents, unit="docs")
        for record in dataset:
            if written >= args.documents:
                break
            if record.get("license_type") != "permissive":
                continue
            pending.add(pool.submit(fetch_record, dict(record)))
            submitted += 1
            if len(pending) < args.in_flight:
                continue
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                item = future.result()
                if item is not None:
                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                    written += 1
                    progress.update(1)
                    if written >= args.documents:
                        break
        while pending and written < args.documents:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                item = future.result()
                if item is not None:
                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                    written += 1
                    progress.update(1)
                    if written >= args.documents:
                        break
        progress.close()
    print(f"Wrote {written:,} documents to {output} ({submitted:,} fetches submitted)")


def _cli_entrypoint() -> None:
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    if os.environ.get("ASTERLM_MATERIALIZER_HARD_EXIT", "1") != "0":
        os._exit(0)


if __name__ == "__main__":
    _cli_entrypoint()
