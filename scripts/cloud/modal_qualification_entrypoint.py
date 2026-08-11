#!/usr/bin/env python3
from __future__ import annotations

import base64
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    encoded = os.environ.pop("ASTERLM_QUALIFICATION_SPEC_B64", "")
    qualification_id = os.environ.get("ASTERLM_QUALIFICATION_ID", "")
    if not encoded or not qualification_id:
        raise RuntimeError("Modal qualification payload is absent")
    spec = json.loads(base64.b64decode(encoded, validate=True))
    if spec.get("data_policy") != "synthetic_or_repository_fixture_only":
        raise RuntimeError("Qualification spec attempted to permit remote corpus access")
    actual_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd="/opt/aster", text=True
    ).strip()
    output = Path("/opt/aster/runs/modal-qualification") / qualification_id
    report_path = output / "summary.json"
    report = {
        "schema_version": 1,
        "qualification_id": qualification_id,
        "status": "running",
        "git_commit": actual_commit,
        "started_unix": time.time(),
        "platform": platform.platform(),
        "python": sys.version,
        "data_policy": spec["data_policy"],
        "checks": [],
    }
    try:
        import torch

        report["torch"] = {
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "device_capability": (
                list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None
            ),
        }
    except (ImportError, RuntimeError) as exc:
        report["torch"] = {"error": f"{type(exc).__name__}: {exc}"}
    identity = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    report["nvidia_smi"] = {
        "returncode": identity.returncode,
        "stdout": identity.stdout.strip(),
        "stderr": identity.stderr.strip(),
    }
    _atomic_json(report_path, report)
    environment = dict(os.environ)
    environment.update(
        {
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "ASTERLM_ALLOW_NETWORK_DATA": "0",
        }
    )
    failed = False
    for check in spec["checks"]:
        command = check.get("command")
        if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
            raise RuntimeError(f"Malformed qualification command: {check}")
        started = time.time()
        try:
            completed = subprocess.run(
                command,
                cwd="/opt/aster",
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=int(check.get("timeout_seconds", 300)),
            )
            returncode = completed.returncode
            stdout = completed.stdout
            stderr = completed.stderr
            status = "passed" if returncode == 0 else "failed"
        except subprocess.TimeoutExpired as exc:
            returncode = 124
            stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            stderr += f"\nqualification timeout after {exc.timeout} seconds"
            status = "timed_out"
        log = output / f"{len(report['checks']):02d}-{check['id']}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(stdout + "\n--- STDERR ---\n" + stderr, encoding="utf-8")
        row = {
            "id": check["id"],
            "command": command,
            "returncode": returncode,
            "duration_seconds": time.time() - started,
            "status": status,
            "log": str(log),
        }
        report["checks"].append(row)
        failed = failed or returncode != 0
        _atomic_json(report_path, report)
    report["status"] = "failed" if failed else "passed"
    report["finished_unix"] = time.time()
    report["duration_seconds"] = report["finished_unix"] - report["started_unix"]
    _atomic_json(report_path, report)
    print("ASTER_MODAL_QUALIFICATION=" + json.dumps(report, sort_keys=True), flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
