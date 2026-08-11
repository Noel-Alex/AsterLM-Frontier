#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from asterlm.cloud import build_modal_launch_plan, dispatch_modal_launch_plan, load_modal_profile


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or dispatch a bounded Modal boost job")
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument(
        "--provider-config", type=Path, default=Path("configs/providers/modal_boost.yaml")
    )
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--execute", action="store_true", help="Actually create a Modal Sandbox; default is dry-run"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    profile = load_modal_profile(args.provider_config, args.profile)
    plan = build_modal_launch_plan(
        contract, profile, root=root, contract_path=args.contract
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = dispatch_modal_launch_plan(plan, execute=args.execute)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
