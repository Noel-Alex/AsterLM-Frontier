#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from asterlm.cloud import build_gcp_launch_plan, dispatch_gcp_launch_plan, load_gcp_profile


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or dispatch a bounded Google Cloud boost job")
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument(
        "--provider-config", type=Path, default=Path("configs/providers/gcp_boost.yaml")
    )
    parser.add_argument("--profile", default="google-credit")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually invoke gcloud; omitted means dry-run only",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    profile = load_gcp_profile(args.provider_config, args.profile)
    plan = build_gcp_launch_plan(contract, profile, root=root, contract_path=args.contract)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    result = dispatch_gcp_launch_plan(plan, execute=args.execute)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
