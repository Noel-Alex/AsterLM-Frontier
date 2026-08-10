#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

VARIANTS: dict[str, dict[str, int]] = {
    # Baseline remains E8/top-2/H704. E4/top-2 doubles rows per expert while
    # preserving active expert width. E4/top-1/H1408 preserves both routed
    # expert capacity and active width while reducing expert/tensor count.
    "grouped-moe-e4k2-kda3": {
        "moe_num_experts": 4,
        "moe_top_k": 2,
        "moe_expert_hidden": 704,
    },
    "grouped-moe-e4k1wide-kda3": {
        "moe_num_experts": 4,
        "moe_top_k": 1,
        "moe_expert_hidden": 1408,
    },
    # This is an implementation-cost isolation control, not an architecture
    # candidate. Removing the always-on shared expert changes model capacity and
    # must never be promoted from a throughput result alone.
    "grouped-moe-e8k2-no-shared-kda3": {
        "moe_num_experts": 8,
        "moe_top_k": 2,
        "moe_expert_hidden": 704,
        "moe_shared_experts": 0,
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate controlled MoE work-granularity configs")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = yaml.safe_load(args.base.read_text(encoding="utf-8"))
    if source.get("model", {}).get("ffn_type") != "moe":
        raise SystemExit("Base config must use ffn_type: moe")
    args.output.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "base": {"path": str(args.base), "sha256": sha256(args.base)},
        "variants": {},
    }
    for name, changes in VARIANTS.items():
        payload = copy.deepcopy(source)
        payload["model"].update(changes)
        target = args.output / f"model-{name}.yaml"
        atomic_text(target, yaml.safe_dump(payload, sort_keys=False))
        manifest["variants"][name] = {
            "path": str(target),
            "sha256": sha256(target),
            "changes": changes,
            "routed_rows_per_expert_at_2048_tokens": 2048 * changes["moe_top_k"] / changes["moe_num_experts"],
            "active_routed_hidden": changes["moe_top_k"] * changes["moe_expert_hidden"],
            "total_routed_hidden": changes["moe_num_experts"] * changes["moe_expert_hidden"],
            "shared_experts": payload["model"].get("moe_shared_experts", 0),
        }
    atomic_text(args.output / "moe-work-granularity-manifest.json", json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
