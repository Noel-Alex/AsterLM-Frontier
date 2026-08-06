#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


TIER_CONFIGS = {
    "pilot": ("configs/corpus/corpus_pilot_500m.yaml", None),
    "18b": ("configs/corpus/corpus_frontier_16b.yaml", "configs/corpus/stack_edu_2p4b.yaml"),
    "50b": ("configs/corpus/corpus_overtrain_50b.yaml", "configs/corpus/stack_edu_6p5b.yaml"),
    "100b": ("configs/corpus/corpus_overtrain_100b.yaml", "configs/corpus/stack_edu_13b.yaml"),
}


def desired_targets(root: Path, tier: str) -> dict[str, int]:
    if tier == "checkpoint":
        return {}
    corpus_file, stack_file = TIER_CONFIGS[tier]
    targets: dict[str, int] = {}
    corpus = yaml.safe_load((root / corpus_file).read_text(encoding="utf-8"))["corpus"]
    for source in corpus["sources"]:
        targets[f"corpus:{source['id']}"] = int(source["target_tokens"])
    if stack_file:
        stack = yaml.safe_load((root / stack_file).read_text(encoding="utf-8"))["stack_edu"]
        for source in stack["languages"]:
            key = str(source["name"]).lower().replace("-", "_")
            targets[f"stack:{key}"] = int(source["target_tokens"])
    return targets


def target_key(path: Path) -> str | None:
    parts = path.parts
    if "corpus-frontier-16b" in parts:
        return f"corpus:{path.parent.name}"
    if "stack-edu-frontier-2p4b" in parts:
        return f"stack:{path.parent.name}"
    return None


def state_rows(root: Path, targets: dict[str, int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("data/**/state.json")):
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            rows.append({"path": str(path), "error": str(exc)})
            continue
        source_state = state.get("source", {})
        checkpoint_target = state.get("target_tokens") or source_state.get("target_tokens")
        has_record_counter = "written" in state or "target_records" in source_state
        target_records = source_state.get("target_records") if has_record_counter else None
        current = state.get("estimated_tokens") if checkpoint_target is not None else state.get("written")
        key = target_key(path)
        desired_target = targets.get(key) if key else None
        target = desired_target if desired_target is not None else (
            checkpoint_target if checkpoint_target is not None else target_records
        )
        pct = (100.0 * float(current) / float(target)) if current is not None and target else None
        complete = bool(current is not None and target is not None and current >= target)
        if target is None:
            complete = bool(state.get("complete", False))
        rows.append(
            {
                "path": str(path),
                "target_key": key,
                "source": state.get("source", {}).get("id") or path.parent.name,
                "current": current,
                "target": target,
                "checkpoint_target": checkpoint_target,
                "percent": pct,
                "seen": state.get("documents_seen", state.get("seen")),
                "written": state.get("documents_written", state.get("written")),
                "shards": state.get("next_shard_index"),
                "retries": state.get("retries", 0),
                "complete": complete,
                "checkpoint_complete": state.get("complete", False),
                "revision": state.get("resolved_revision"),
                "last_reason": state.get("last_checkpoint_reason"),
            }
        )
    return rows


def human(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        for suffix, scale in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
            if abs(value) >= scale:
                return f"{value / scale:.2f}{suffix}"
        return f"{value:,.0f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize AsterLM data-download checkpoints")
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--tier",
        choices=["checkpoint", *TIER_CONFIGS],
        default="100b",
        help="Show progress against this campaign target, not merely the last completed pilot cap",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    targets = desired_targets(root, args.tier)
    rows = state_rows(root, targets)
    if args.json:
        print(json.dumps({"tier": args.tier, "targets": targets, "rows": rows}, indent=2, default=str))
        return
    if not rows:
        print("No data/**/state.json checkpoints found.")
        return

    campaign_rows = [row for row in rows if row.get("target_key") in targets]
    if campaign_rows:
        current = sum(int(row.get("current") or 0) for row in campaign_rows)
        target = sum(targets.values())
        print(
            f"PRETRAIN TIER {args.tier.upper()}: {human(current)}/{human(target)} "
            f"({100.0 * current / target:.2f}%). Completed pilot caps do not mean this tier is complete."
        )
    print(f"{'SOURCE':34} {'PROGRESS':>19} {'PCT':>8} {'SHARDS':>7} {'RETRY':>6} {'DONE':>5}  LAST")
    for row in rows:
        source = str(row.get("source", "-"))
        if len(source) > 34:
            source = source[:31] + "..."
        pct = f"{row['percent']:.2f}%" if row.get("percent") is not None else "-"
        target = row.get("target")
        progress = (
            f"{human(row.get('current'))}/{human(target)}"
            if target is not None
            else f"{human(row.get('current'))}/ALL"
        )
        print(
            f"{source:34} {progress:>19} {pct:>8} {str(row.get('shards', '-')):>7} "
            f"{str(row.get('retries', 0)):>6} {str(bool(row.get('complete'))):>5}  {row.get('last_reason') or '-'}"
        )


if __name__ == "__main__":
    main()
