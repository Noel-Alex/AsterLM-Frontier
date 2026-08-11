#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from asterlm.cloud import load_gcp_profile
from asterlm.cloud.gcp_cache import build_gcp_cache_stage_plan, execute_gcp_cache_stage


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan or stage one sealed clean corpus into Google Cloud Storage"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--profile", default="google-credit")
    parser.add_argument(
        "--provider-config", type=Path, default=Path("configs/providers/gcp_boost.yaml")
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-local-hashes", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Upload sealed artifacts to Cloud Storage; never creates a VM or GPU",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    profile = load_gcp_profile(args.provider_config, args.profile)
    plan = build_gcp_cache_stage_plan(
        args.manifest,
        profile,
        root=root,
        verify_local_hashes=args.verify_local_hashes or args.execute,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = execute_gcp_cache_stage(plan) if args.execute else {"status": "dry_run", "plan": plan}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
