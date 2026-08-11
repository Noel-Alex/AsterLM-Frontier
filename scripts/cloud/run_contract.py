#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and execute an immutable Aster remote contract"
    )
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    if contract.get("provider") != os.environ.get("ASTERLM_REMOTE_PROVIDER"):
        raise RuntimeError("Remote provider identity does not match contract")
    command = contract.get("command")
    if not isinstance(command, list) or command[:3] != [
        "python",
        "scripts/studio_train.py",
        "--mode",
    ]:
        raise RuntimeError("Contract command is not an approved Aster training entrypoint")
    for item in contract.get("inputs", {}).values():
        path = Path(item["path"])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != item["sha256"]:
            raise RuntimeError(f"Contract input hash mismatch: {path}")
    completed = subprocess.run(command, check=False)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
