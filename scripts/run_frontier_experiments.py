#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def run_case(case: dict[str, Any], output: Path, steps: int) -> dict[str, Any]:
    name = case["name"]
    destination = output / f"{name}.json"
    command = [
        sys.executable,
        "scripts/profile_training.py",
        "--model",
        case["model"],
        "--train-config",
        "configs/train/probe_memory_matrix.yaml",
        "--sequence",
        str(case["sequence"]),
        "--steps",
        str(steps),
        "--warmup",
        "1",
        "--optimizer",
        case["optimizer"],
        "--json",
        str(destination),
    ]
    if case.get("activation_offload"):
        command.append("--activation-offload")
    if case.get("precision"):
        command.extend(["--precision", case["precision"]])
    print("\n$", " ".join(command), flush=True)
    completed = subprocess.run(command, check=False)
    record: dict[str, Any] = {**case, "returncode": completed.returncode, "result": str(destination)}
    if destination.exists():
        payload = json.loads(destination.read_text(encoding="utf-8"))
        record["status"] = payload.get("status")
        record["summary"] = payload.get("summary")
        record["error"] = payload.get("error")
    else:
        record["status"] = "missing_result"
    return record


def score(record: dict[str, Any]) -> tuple:
    ok = record.get("status") == "ok"
    summary = record.get("summary") or {}
    peak = ((summary.get("final_memory") or {}).get("peak_allocated_gib") or 999.0)
    speed = summary.get("median_tokens_per_second") or 0.0
    return (not ok, peak > 11.25, -record.get("sequence", 0), -speed, peak)


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated VRAM/precision/optimizer experiment matrix")
    parser.add_argument("--output", default="runs/frontier-experiments")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    if args.mode == "quick":
        cases = [
            {
                "model": "configs/model/aster_moe_probe_674m_a367m.yaml",
                "sequence": 4096,
                "optimizer": "apollo_mini",
                "activation_offload": False,
                "precision": "amp",
            },
            {
                "model": "configs/model/aster_moe_frontier_893m_a484m.yaml",
                "sequence": 4096,
                "optimizer": "apollo_mini",
                "activation_offload": True,
                "precision": "amp",
            },
            {
                "model": "configs/model/aster_dense_challenger_666m.yaml",
                "sequence": 4096,
                "optimizer": "apollo_mini",
                "activation_offload": True,
                "precision": "amp",
            },
            {
                "model": "configs/model/aster_moe_frontier_893m_a484m.yaml",
                "sequence": 8192,
                "optimizer": "torchao_cpu_offload_adamw",
                "activation_offload": True,
                "precision": "amp",
            },
            {
                "model": "configs/model/aster_moe_target_1p51b_a623m.yaml",
                "sequence": 2048,
                "optimizer": "torchao_adamw4bit",
                "activation_offload": True,
                "precision": "amp",
            },
            {
                "model": "configs/model/aster_moe_frontier_893m_loqt.yaml",
                "sequence": 8192,
                "optimizer": "torchao_adamw8bit",
                "activation_offload": True,
                "precision": "amp",
            },
        ]
    else:
        models = [
            "configs/model/aster_moe_probe_674m_a367m.yaml",
            "configs/model/aster_moe_frontier_893m_a484m.yaml",
            "configs/model/aster_dense_challenger_666m.yaml",
            "configs/model/aster_moe_target_1p51b_a623m.yaml",
            "configs/model/aster_moe_stretch_1p90b_a769m.yaml",
            "configs/model/aster_moe_frontier_893m_loqt.yaml",
        ]
        sequences = [2048, 4096, 8192, 16384]
        optimizers = ["apollo_mini", "torchao_adamw4bit", "torchao_cpu_offload_adamw"]
        cases = [
            {
                "model": model,
                "sequence": sequence,
                "optimizer": optimizer,
                "activation_offload": offload,
                "precision": "amp",
            }
            for model, sequence, optimizer, offload in itertools.product(
                models, sequences, optimizers, [False, True]
            )
        ]
        # FP8 is a separate shape-identical experiment on TE modules.
        for sequence in [2048, 4096, 8192]:
            cases.append(
                {
                    "model": "configs/model/aster_moe_frontier_893m_fp8.yaml",
                    "sequence": sequence,
                    "optimizer": "apollo_mini",
                    "activation_offload": True,
                    "precision": "transformer_engine_fp8",
                }
            )

    for index, case in enumerate(cases):
        case["name"] = (
            f"{index:03d}-{Path(case['model']).stem}-s{case['sequence']}-"
            f"{case['optimizer']}-off{int(case['activation_offload'])}-{case['precision']}"
        )
    index_path = output / "index.json"
    records: list[dict[str, Any]] = []
    for case in cases:
        records.append(run_case(case, output, args.steps))
        index_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    ranked = sorted(records, key=score)
    (output / "ranked.json").write_text(json.dumps(ranked, indent=2), encoding="utf-8")
    print("\nTop fitting cases:")
    for item in ranked[:10]:
        print(item["name"], item.get("status"), item.get("summary"))
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
