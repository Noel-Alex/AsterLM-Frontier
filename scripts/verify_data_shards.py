#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm


def verify(path: Path, chunk_bytes: int = 8 * 2**20) -> dict[str, Any]:
    import zstandard as zstd

    started = time.time()
    decompressed = 0
    lines = 0
    try:
        with path.open("rb") as raw, zstd.ZstdDecompressor().stream_reader(raw) as reader:
            buffer = b""
            while True:
                chunk = reader.read(chunk_bytes)
                if not chunk:
                    break
                decompressed += len(chunk)
                lines += chunk.count(b"\n")
                buffer = chunk[-1:]
        return {
            "path": str(path),
            "status": "ok",
            "compressed_bytes": path.stat().st_size,
            "decompressed_bytes": decompressed,
            "records": lines,
            "seconds": time.time() - started,
            "ends_with_newline": buffer in (b"", b"\n"),
        }
    except Exception as exc:
        return {
            "path": str(path),
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "seconds": time.time() - started,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Read every zstd frame and verify AsterLM corpus shards")
    parser.add_argument("paths", nargs="*", default=["data"])
    parser.add_argument("--output", default="data/shard_verification.json")
    parser.add_argument("--only-last", action="store_true", help="Verify only the newest shard in each source directory")
    args = parser.parse_args()

    files: list[Path] = []
    for item in args.paths:
        path = Path(item)
        if path.is_file() and path.suffix == ".zst":
            files.append(path)
        elif path.exists():
            if args.only_last:
                for parent in {p.parent for p in path.rglob("*.jsonl.zst")}:
                    candidates = sorted(parent.glob("*.jsonl.zst"))
                    if candidates:
                        files.append(candidates[-1])
            else:
                files.extend(path.rglob("*.jsonl.zst"))
    files = sorted(set(files))
    results: list[dict[str, Any]] = []
    for path in tqdm(files, desc="verifying shards", dynamic_ncols=True):
        result = verify(path)
        results.append(result)
        if result["status"] != "ok":
            tqdm.write(f"CORRUPT: {path}: {result['error']}")

    report = {
        "files": len(results),
        "failures": sum(row["status"] != "ok" for row in results),
        "results": results,
        "finished_at_unix": time.time(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"verified={report['files']} failures={report['failures']} report={output}")
    if report["failures"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
