#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
from pathlib import Path

import yaml


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    if not files:
        raise RuntimeError(f"Contract directory contains no files: {path}")
    for candidate in files:
        if candidate.is_symlink():
            raise RuntimeError(f"Contract directories may not contain symlinks: {candidate}")
        relative = candidate.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(candidate.stat().st_size.to_bytes(8, "big"))
        digest.update(bytes.fromhex(_file_sha256(candidate)))
    return digest.hexdigest()


def _replace_pair(command: list[str], option: str, value: str) -> None:
    try:
        index = command.index(option)
    except ValueError as exc:
        raise RuntimeError(f"Remote contract is missing {option}") from exc
    if index + 1 >= len(command):
        raise RuntimeError(f"Remote contract has no value for {option}")
    command[index + 1] = value


def _run_training(command: list[str]) -> int:
    """Run one trainer process group and forward container stop signals to it."""

    child = subprocess.Popen(command, start_new_session=True)

    def forward_stop(signum: int, _frame: object) -> None:
        if child.poll() is None:
            os.killpg(child.pid, signum)

    old_int = signal.signal(signal.SIGINT, forward_stop)
    old_term = signal.signal(signal.SIGTERM, forward_stop)
    try:
        return child.wait()
    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)


def _materialize_hub_checkpoint(
    contract: dict, *, key: str, destination: Path
) -> Path:
    checkpoint_record = contract.get(key)
    if not isinstance(checkpoint_record, dict):
        raise TypeError(f"Contract uses Hub checkpoint sentinel without {key} metadata")
    from huggingface_hub import snapshot_download

    path_in_repo = str(checkpoint_record["path"]).strip("/")
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=str(checkpoint_record["repo_id"]),
        repo_type="model",
        revision=str(checkpoint_record["revision"]),
        allow_patterns=[f"{path_in_repo}/**"],
        local_dir=destination,
    )
    checkpoint = destination / Path(path_in_repo)
    if not (checkpoint / "checkpoint_manifest.json").is_file():
        raise RuntimeError(f"Hub resume did not materialize a checkpoint: {path_in_repo}")
    from asterlm.training.checkpoint import verify_checkpoint

    verify_checkpoint(checkpoint)
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and execute an immutable Aster remote contract"
    )
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    if contract.get("provider") != os.environ.get("ASTERLM_REMOTE_PROVIDER"):
        raise RuntimeError("Remote provider identity does not match contract")
    command = list(contract.get("command") or [])
    if not isinstance(command, list) or command[:3] != [
        "python",
        "scripts/studio_train.py",
        "--mode",
    ]:
        raise RuntimeError("Contract command is not an approved Aster training entrypoint")
    for item in contract.get("inputs", {}).values():
        path = Path(item["path"])
        digest = _tree_sha256(path) if item.get("kind") == "directory" else _file_sha256(path)
        if digest != item["sha256"]:
            raise RuntimeError(f"Contract input hash mismatch: {path}")
    if contract.get("gpu"):
        from asterlm.artifacts import atomic_write_json, atomic_write_text
        from asterlm.cloud.execution_profiles import resolve_gpu_execution_profile

        train_option = command.index("--train")
        source_train = Path(command[train_option + 1])
        resolved, identity = resolve_gpu_execution_profile(
            source_train,
            str(contract["gpu"]),
            profiles_path="configs/providers/gpu_execution_profiles.yaml",
        )
        resolved_path = Path("/run/aster/resolved_train.yaml")
        atomic_write_text(resolved_path, yaml.safe_dump(resolved, sort_keys=False))
        identity_path = Path("/run/aster/remote_execution_profile.json")
        atomic_write_json(identity_path, identity)
        os.environ["ASTERLM_REMOTE_EXECUTION_PROFILE"] = str(identity_path)
        os.environ["ASTERLM_REMOTE_GPU"] = str(contract["gpu"])
        command[train_option + 1] = str(resolved_path)
    if "__ASTER_HUB_RESUME__" in command:
        resume_root = Path(os.environ.get("ASTERLM_HUB_RESUME_ROOT", "/var/cache/aster/hub-resume"))
        checkpoint = _materialize_hub_checkpoint(
            contract,
            key="resume_hub",
            destination=resume_root / contract["contract_id"] / "resume",
        )
        _replace_pair(command, "--resume", str(checkpoint))
    if "__ASTER_HUB_INIT__" in command:
        resume_root = Path(os.environ.get("ASTERLM_HUB_RESUME_ROOT", "/var/cache/aster/hub-resume"))
        checkpoint = _materialize_hub_checkpoint(
            contract,
            key="init_hub",
            destination=resume_root / contract["contract_id"] / "init",
        )
        _replace_pair(command, "--init-checkpoint", str(checkpoint))
    raise SystemExit(_run_training(command))


if __name__ == "__main__":
    main()
