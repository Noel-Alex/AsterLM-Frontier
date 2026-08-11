#!/usr/bin/env python3
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    encoded = os.environ.pop("ASTERLM_REMOTE_CONTRACT_B64", "")
    if not encoded:
        raise RuntimeError("Modal contract payload is absent")
    contract_bytes = base64.b64decode(encoded, validate=True)
    contract = json.loads(contract_bytes)
    if contract.get("provider") != "modal":
        raise RuntimeError("Modal entrypoint received a non-Modal contract")
    actual_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd="/opt/aster", text=True
    ).strip()
    if actual_commit != contract.get("git_commit"):
        raise RuntimeError(
            f"Image source mismatch: contract={contract.get('git_commit')} image={actual_commit}"
        )
    run_root = Path("/run/aster")
    run_root.mkdir(parents=True, exist_ok=True)
    contract_path = run_root / "contract.json"
    contract_path.write_bytes(contract_bytes)
    os.execv(
        sys.executable,
        [
            sys.executable,
            "scripts/cloud/run_contract.py",
            "--contract",
            str(contract_path),
        ],
    )


if __name__ == "__main__":
    main()
