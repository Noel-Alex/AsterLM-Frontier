#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from studio.research_archive import ResearchArchive

    parser = argparse.ArgumentParser(description="Index all AsterLM research artifacts for Studio.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--database", type=Path)
    args = parser.parse_args()
    archive = ResearchArchive(args.root, args.database)
    print(json.dumps(archive.reindex(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
