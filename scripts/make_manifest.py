#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def describe_file(value: str | None) -> dict[str, Any] | None:
    if not value:
        return None
    path = Path(value).resolve()
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size if path.exists() else None}


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a reproducibility manifest for an AsterLM run")
    parser.add_argument("--model", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", default="artifacts/run_manifest.json")
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    try:
        import torch

        torch_info: dict[str, Any] = {
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cudnn": torch.backends.cudnn.version(),
        }
        if torch.cuda.is_available():
            torch_info["devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            torch_info["capabilities"] = [list(torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())]
    except Exception as exc:
        torch_info = {"error": repr(exc)}

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "note": args.note,
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(command_output(["git", "status", "--porcelain"])),
        "python": {"version": sys.version, "executable": sys.executable},
        "platform": {"platform": platform.platform(), "machine": platform.machine()},
        "torch": torch_info,
        "packages": {
            name: package_version(name)
            for name in ["asterlm", "fla-core", "transformers", "datasets", "tokenizers", "torchao", "safetensors"]
        },
        "nvidia_smi": command_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]
        ),
        "inputs": {
            "model_config": describe_file(args.model),
            "train_config": describe_file(args.train),
            "data_config": describe_file(args.data),
            "tokenizer": describe_file(args.tokenizer),
            "checkpoint": describe_file(args.checkpoint),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(output.resolve())


if __name__ == "__main__":
    main()
