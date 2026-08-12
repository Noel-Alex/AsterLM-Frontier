#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from asterlm.artifacts import atomic_write_json
from asterlm.experiments.quality_analysis import analyze_quality_campaign


def _format(value: object, digits: int = 4) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze matched architecture learning curves")
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze_quality_campaign(args.campaign)
    output = args.output or args.campaign.with_name("quality-analysis.json")
    atomic_write_json(output, result)
    lines = [
        "# Architecture quality analysis",
        "",
        f"Status: **{result['analysis_status']}** ({result['complete_runs']}/{result['expected_runs']} runs)",
        "",
        "| Candidate | Final loss | Equal-wall loss | Equal-FLOP loss | tok/s | GPU | VRAM GiB |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for identity, row in result["candidates"].items():
        lines.append(
            f"| {identity} | {_format(row['final_eval_loss_mean'])} | "
            f"{_format(row['equal_wall_loss_mean'])} | "
            f"{_format(row['equal_active_flops_loss_mean'])} | "
            f"{_format(row['median_training_tokens_per_second'], 0)} | "
            f"{_format(row['mean_gpu_util_percent'], 1)} | {_format(row['peak_vram_gib'], 2)} |"
        )
    markdown = output.with_suffix(".md")
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(output), "markdown": str(markdown), **result["selection"]}, indent=2))


if __name__ == "__main__":
    main()
