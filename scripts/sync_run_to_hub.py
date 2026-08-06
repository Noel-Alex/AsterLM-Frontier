#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from asterlm.training.hub import HubRunSync


def main() -> None:
    parser = argparse.ArgumentParser(description="Retry or manually trigger an AsterLM Hugging Face backup")
    parser.add_argument("--run", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--checkpoint", default=None, help="Defaults to the run's latest.txt target")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--model-only", action="store_true", help="Exclude trainer_state.pt/optimizer state")
    parser.add_argument("--reason", default="manual-sync")
    args = parser.parse_args()

    root = Path(args.run)
    checkpoint = Path(args.checkpoint) if args.checkpoint else Path(
        (root / "latest.txt").read_text(encoding="utf-8").strip()
    )
    manifest_path = checkpoint / "checkpoint_manifest.json"
    manifest = {}
    if manifest_path.exists():
        import json

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sync = HubRunSync(
        repo_id=args.repo,
        private=not args.public,
        revision=args.revision,
        include_optimizer=not args.model_only,
    )
    result = sync.sync(
        output_dir=root,
        checkpoint=checkpoint,
        reason=args.reason,
        step=int(manifest.get("step", 0)),
        tokens_seen=int(manifest.get("tokens_seen", 0)),
    )
    print(result)


if __name__ == "__main__":
    main()
