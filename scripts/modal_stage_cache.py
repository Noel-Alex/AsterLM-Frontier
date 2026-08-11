#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from asterlm.cloud.modal import load_modal_profile
from asterlm.cloud.modal_cache import build_modal_cache_stage_plan, execute_modal_cache_stage


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan or stage one sealed clean corpus into a persistent Modal Volume"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument(
        "--provider-config", type=Path, default=Path("configs/providers/modal_boost.yaml")
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--verify-local-hashes",
        action="store_true",
        help="Rehash every sealed artifact in dry-run mode (always enabled with --execute)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform the Volume-only upload; never creates a Sandbox or GPU",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    profile = load_modal_profile(args.provider_config, args.profile)
    plan = build_modal_cache_stage_plan(
        args.manifest,
        root=root,
        profile_alias=profile.alias,
        modal_environment=profile.modal_environment,
        volume_name=profile.dataset_volume,
        volume_version=profile.volume_version,
        verify_local_hashes=args.verify_local_hashes or args.execute,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.execute:
        os.environ.setdefault("MODAL_ENVIRONMENT", profile.modal_environment)
        result = execute_modal_cache_stage(plan)
    else:
        result = {"status": "dry_run", "plan": plan}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
