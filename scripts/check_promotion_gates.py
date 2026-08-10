#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from asterlm.experiments import evaluate_promotion_gates


def main() -> None:
    parser = argparse.ArgumentParser(description="Enforce the AsterLM final-run promotion lock")
    parser.add_argument(
        "--gates", type=Path, default=Path("configs/experiments/promotion_gates.yaml")
    )
    parser.add_argument(
        "--require-ready", action="store_true", help="Exit non-zero unless every required gate passed"
    )
    args = parser.parse_args()
    decision = evaluate_promotion_gates(args.gates)
    print(json.dumps(decision.manifest(), indent=2))
    if args.require_ready and not decision.ready:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
