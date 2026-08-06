#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def last_jsonl(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    last = None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                continue
    return last


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize an AsterLM run and its protected milestones")
    parser.add_argument("run")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    root = Path(args.run)
    if not root.exists():
        raise SystemExit(f"Missing run directory: {root}")

    checkpoints = []
    for path in sorted(root.glob("checkpoint-*")):
        manifest = read_json(path / "checkpoint_manifest.json") or {}
        checkpoints.append(
            {
                "name": path.name,
                "permanent": (path / "KEEP").exists(),
                **manifest,
            }
        )
    usage = shutil.disk_usage(root)
    payload = {
        "run": str(root.resolve()),
        "last_metric": last_jsonl(root / "metrics.jsonl"),
        "latest_checkpoint": (
            (root / "latest.txt").read_text(encoding="utf-8").strip()
            if (root / "latest.txt").exists()
            else None
        ),
        "hub_sync": read_json(root / "hub_sync_state.json"),
        "checkpoints": checkpoints,
        "permanent_milestones": [item for item in checkpoints if item["permanent"]],
        "free_disk_gib": usage.free / 2**30,
    }
    if args.json:
        print(json.dumps(payload, indent=2, default=str))
        return
    metric = payload["last_metric"] or {}
    print(f"run: {payload['run']}")
    print(
        f"step={metric.get('step', 'n/a')} tokens={metric.get('tokens_seen', 'n/a')} "
        f"loss={metric.get('loss', metric.get('eval_loss', 'n/a'))}"
    )
    print(f"latest: {payload['latest_checkpoint']}")
    print(f"checkpoints: {len(checkpoints)}; permanent: {len(payload['permanent_milestones'])}")
    print(f"hub sync: {payload['hub_sync'] or 'not configured/not yet synced'}")
    print(f"free disk: {payload['free_disk_gib']:.1f} GiB")


if __name__ == "__main__":
    main()
