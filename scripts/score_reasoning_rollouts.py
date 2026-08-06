#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import torch

from asterlm.reasoning import RLVRConfig, compute_group_advantages, score_completion
from asterlm.reasoning.io import atomic_write_jsonl, iter_json_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply deterministic verifiers to reasoning rollouts")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reasoning", default="configs/reasoning/rlvr_laptop_gspo.yaml")
    args = parser.parse_args()

    config = RLVRConfig.from_yaml(args.reasoning)
    records = list(iter_json_records(args.input))
    if not records:
        raise ValueError("No rollout records found")
    grouped: dict[int, list[int]] = defaultdict(list)
    rewards = []
    for index, record in enumerate(records):
        breakdown = score_completion(record, str(record.get("completion", "")), config)
        record["reward"] = breakdown.total
        record["reward_breakdown"] = breakdown.to_dict()
        rewards.append(breakdown.total)
        grouped[int(record["group_id"])].append(index)

    reward_tensor = torch.tensor(rewards, dtype=torch.float32)
    group_ids = torch.tensor([int(record["group_id"]) for record in records], dtype=torch.long)
    advantages, group_stds = compute_group_advantages(
        reward_tensor,
        group_ids,
        algorithm=config.algorithm,
        eps=config.advantage_epsilon,
        normalize_std=config.normalize_group_std,
    )
    unique_groups = sorted(grouped)
    group_std_map = {group: float(std) for group, std in zip(unique_groups, group_stds, strict=True)}
    usable = 0
    for index, record in enumerate(records):
        std = group_std_map[int(record["group_id"])]
        record["advantage"] = float(advantages[index])
        record["group_reward_std"] = std
        record["usable_group"] = not config.dynamic_sampling or std >= config.min_group_reward_std
        usable += int(record["usable_group"])

    atomic_write_jsonl(args.output, records)
    summary = {
        "records": len(records),
        "groups": len(grouped),
        "usable_records": usable,
        "mean_reward": float(reward_tensor.mean()),
        "correct_rate": sum(float(r["reward_breakdown"]["correctness"]) for r in records) / len(records),
        "format_rate": sum(float(r["reward_breakdown"]["format"]) for r in records) / len(records),
        "zero_variance_groups": sum(std < config.min_group_reward_std for std in group_std_map.values()),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
