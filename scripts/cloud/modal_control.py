#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect or stop one Aster Modal Sandbox")
    parser.add_argument("--sandbox-id", required=True)
    parser.add_argument("--mode", choices=("status", "graceful", "terminate"), required=True)
    args = parser.parse_args()
    if not os.environ.get("MODAL_PROFILE"):
        raise RuntimeError("MODAL_PROFILE is required for isolated workspace control")

    # Import only after the caller selects the owning Modal profile.
    import modal

    sandbox = modal.Sandbox.from_id(args.sandbox_id)
    if args.mode == "status":
        returncode = sandbox.poll()
        result = {
            "status": "running" if returncode is None else "exited",
            "sandbox_id": args.sandbox_id,
            "returncode": returncode,
        }
    elif args.mode == "graceful":
        # Studio handles SIGTERM only at an optimizer-update boundary, writes a
        # full checkpoint, verifies its Hub upload, and then lets the main process
        # exit. Modal consequently tears down the Sandbox without idle billing.
        signaler = sandbox.exec(
            "bash",
            "-lc",
            "pkill -TERM -f '[s]cripts/studio_train.py'",
        )
        returncode = signaler.wait()
        if returncode != 0:
            raise RuntimeError("The Aster training process was not running in this Sandbox")
        result = {
            "status": "graceful_stop_requested",
            "sandbox_id": args.sandbox_id,
            "checkpoint_policy": "optimizer-boundary checkpoint plus verified Hub upload",
        }
    else:
        returncode = sandbox.terminate(wait=True)
        result = {
            "status": "terminated",
            "sandbox_id": args.sandbox_id,
            "returncode": returncode,
            "warning": "Only checkpoints completed before termination are recoverable",
        }
    print("ASTER_MODAL_CONTROL_RESULT=" + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
