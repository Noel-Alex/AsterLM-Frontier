#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from asterlm.experiments import load_architecture_campaign, materialize_architecture_campaign


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and materialize the controlled architecture campaign"
    )
    parser.add_argument(
        "--campaign", type=Path, default=Path("configs/experiments/architecture_campaign.yaml")
    )
    parser.add_argument("--output", type=Path, default=Path("runs/architecture-campaign/configs"))
    args = parser.parse_args()
    campaign = load_architecture_campaign(args.campaign, repo_root=Path.cwd())
    manifest = materialize_architecture_campaign(campaign, args.output)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
