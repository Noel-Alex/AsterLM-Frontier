#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = ROOT / "docs" / "research" / "findings.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(description="Append a durable finding to the AsterLM research ledger.")
    parser.add_argument("--id", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--status", default="observed")
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--evidence", action="append", default=[])
    parser.add_argument("--supersedes")
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    args = parser.parse_args()
    finding = {
        "id": args.id,
        "created_utc": datetime.now(UTC).isoformat(),
        "title": args.title,
        "summary": args.summary,
        "status": args.status,
        "tags": args.tag,
        "evidence": args.evidence,
    }
    if args.supersedes:
        finding["supersedes"] = args.supersedes
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(finding, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(args.ledger, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(descriptor, line)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    print(json.dumps(finding, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
