#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter
from pathlib import Path

from asterlm.data.quality import entropy_bits, quality_decision
from clean_corpus import iter_records, nested_get


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * q))]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample and audit a local materialized corpus")
    parser.add_argument("--input", required=True)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--sample", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    reservoir: list[tuple[dict, Path]] = []
    total = 0
    for record, path in iter_records(Path(args.input)):
        total += 1
        if len(reservoir) < args.sample:
            reservoir.append((record, path))
        else:
            index = rng.randrange(total)
            if index < args.sample:
                reservoir[index] = (record, path)

    chars: list[float] = []
    words: list[float] = []
    entropy: list[float] = []
    rejection: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    pii = 0
    for record, path in reservoir:
        value = nested_get(record, args.text_field)
        if value is None:
            rejection["missing_text"] += 1
            continue
        decision = quality_decision(str(value))
        rejection[decision.reason] += 1
        chars.append(decision.metrics.get("chars", 0.0))
        words.append(decision.metrics.get("words", 0.0))
        pii += int(decision.metrics.get("pii_count", 0.0))
        entropy.append(entropy_bits(decision.normalized_text))
        sources[str(record.get("_source_id") or record.get("_dataset") or path.parent.name)] += 1

    report = {
        "input": args.input,
        "total_records_seen": total,
        "sample_size": len(reservoir),
        "quality_decisions": dict(rejection),
        "sources": dict(sources.most_common()),
        "detected_pii_items_before_redaction": pii,
        "chars": {
            "mean": statistics.fmean(chars) if chars else 0.0,
            "p50": quantile(chars, 0.5),
            "p90": quantile(chars, 0.9),
            "p99": quantile(chars, 0.99),
        },
        "words": {
            "mean": statistics.fmean(words) if words else 0.0,
            "p50": quantile(words, 0.5),
            "p90": quantile(words, 0.9),
        },
        "character_entropy_bits": {
            "mean": statistics.fmean(entropy) if entropy else 0.0,
            "p10": quantile(entropy, 0.1),
            "p90": quantile(entropy, 0.9),
        },
    }
    output = Path(args.output) if args.output else Path(args.input) / "audit_report.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
