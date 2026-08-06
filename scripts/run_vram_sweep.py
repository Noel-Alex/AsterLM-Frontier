#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run each VRAM probe in a clean process")
    parser.add_argument("--output", default="runs/vram-sweep")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--optimizer", default="apollo_mini")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    models = [
        "configs/model/aster_moe_probe_674m_a367m.yaml",
        "configs/model/aster_moe_target_1p51b_a623m.yaml",
        "configs/model/aster_moe_stretch_1p90b_a769m.yaml",
    ]
    sequences = [1024, 2048, 4096, 8192]
    index = []
    for model in models:
        model_name = Path(model).stem
        for sequence in sequences:
            destination = output / f"{model_name}-s{sequence}.json"
            command = [
                sys.executable,
                "scripts/profile_training.py",
                "--model",
                model,
                "--train-config",
                "configs/train/probe_apollo_2k.yaml",
                "--sequence",
                str(sequence),
                "--accum",
                "1",
                "--steps",
                str(args.steps),
                "--warmup",
                "1",
                "--optimizer",
                args.optimizer,
                "--json",
                str(destination),
            ]
            print(" ".join(command), flush=True)
            completed = subprocess.run(command, check=False)
            record = {"model": model, "sequence": sequence, "returncode": completed.returncode, "json": str(destination)}
            if destination.exists():
                payload = json.loads(destination.read_text(encoding="utf-8"))
                record["status"] = payload.get("status")
                record["summary"] = payload.get("summary")
                record["error"] = payload.get("error")
            index.append(record)
            (output / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"Sweep index: {output / 'index.json'}")


if __name__ == "__main__":
    main()
