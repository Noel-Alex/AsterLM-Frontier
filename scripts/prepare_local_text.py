#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize local text files into JSONL for AsterLM")
    parser.add_argument("input", nargs="+", help="Files or directories")
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-chars", type=int, default=64)
    args = parser.parse_args()

    files: list[Path] = []
    for raw in args.input:
        path = Path(raw)
        files.extend(sorted(path.rglob("*")) if path.is_dir() else [path])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output.open("w", encoding="utf-8") as handle:
        for path in files:
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if len(text) < args.min_chars:
                continue
            handle.write(json.dumps({"text": text, "source": str(path)}, ensure_ascii=False) + "\n")
            written += 1
    print(f"Wrote {written} records to {output}")


if __name__ == "__main__":
    main()
