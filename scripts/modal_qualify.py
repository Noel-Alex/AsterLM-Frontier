#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from asterlm.cloud import (
    build_modal_qualification_plan,
    dispatch_modal_launch_plan,
    load_modal_profile,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan one consolidated Modal qualification Sandbox")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--timeout-minutes", type=int, default=30)
    parser.add_argument("--estimated-spend-usd", type=float, required=True)
    parser.add_argument(
        "--provider-config", type=Path, default=Path("configs/providers/modal_boost.yaml")
    )
    parser.add_argument(
        "--spec", type=Path, default=Path("configs/providers/modal_qualification.json")
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    profile = load_modal_profile(args.provider_config, args.profile)
    plan = build_modal_qualification_plan(
        args.spec,
        profile,
        root=root,
        gpu=args.gpu,
        timeout_minutes=args.timeout_minutes,
        estimated_spend_usd=args.estimated_spend_usd,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(dispatch_modal_launch_plan(plan, execute=args.execute), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
