#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import os
from pathlib import Path

from asterlm.cuda_allocator import cuda_allocator_environment

MAX_CACHED_MANIFEST_BYTES = 64 * 1024 * 1024


async def _verify_dataset_cache(volume, requirement: dict[str, str]) -> dict[str, str | int]:
    """Verify the exact cache commit marker without allocating a Sandbox."""

    remote_path = str(requirement.get("volume_path") or "")
    expected = str(requirement.get("sha256") or "")
    if not remote_path or len(expected) != 64:
        raise RuntimeError("Modal launch plan has no valid dataset-cache requirement")
    digest = hashlib.sha256()
    size = 0
    try:
        async for chunk in volume.read_file(remote_path):
            size += len(chunk)
            if size > MAX_CACHED_MANIFEST_BYTES:
                raise RuntimeError("Cached dataset manifest exceeds the safety limit")
            digest.update(chunk)
    except Exception as exc:
        raise RuntimeError(
            f"Required dataset cache manifest is absent or unreadable: {remote_path}"
        ) from exc
    observed = digest.hexdigest()
    if observed != expected:
        raise RuntimeError(
            "Required dataset cache manifest hash mismatch: "
            f"expected={expected} observed={observed} path={remote_path}"
        )
    return {"volume_path": remote_path, "sha256": observed, "size_bytes": size}


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit one immutable Aster contract to Modal")
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if plan.get("provider") != "modal" or plan.get("blockers"):
        raise RuntimeError("Refusing a non-Modal or blocked launch plan")
    if os.environ.get("MODAL_PROFILE") != plan["profile_alias"]:
        raise RuntimeError("MODAL_PROFILE does not match the launch plan")

    payload_path = Path(plan["contract_path"])
    payload_bytes = payload_path.read_bytes()
    if hashlib.sha256(payload_bytes).hexdigest() != plan["contract_sha256"]:
        raise RuntimeError("Modal payload changed after launch-plan creation")

    # Import only after MODAL_PROFILE has been isolated in this subprocess.
    import modal
    from modal.exception import Error as ModalError

    environment_name = plan["modal_environment"]
    job_kind = str(plan.get("job_kind") or "training")
    dataset_mount = "/opt/aster/data"
    cache_verification = None
    dataset_volume = None
    if job_kind == "training":
        dataset_volume = modal.Volume.from_name(
            plan["volumes"][dataset_mount],
            environment_name=environment_name,
            create_if_missing=False,
            version=int(plan["volume_version"]),
        )
        requirement = (plan.get("cache_policy") or {}).get("required_dataset_manifest")
        if not isinstance(requirement, dict):
            raise RuntimeError("Modal launch plan has no required dataset-cache manifest")
        cache_verification = asyncio.run(_verify_dataset_cache(dataset_volume, requirement))
    elif job_kind != "qualification":
        raise RuntimeError(f"Unsupported Modal job kind: {job_kind}")

    image_spec = plan["image"]
    commit = image_spec["git_commit"]
    repository = image_spec["repository_url"]
    image = (
        modal.Image.from_registry(image_spec["base"])
        .apt_install("git")
        .run_commands(
            f"git clone --filter=blob:none --no-checkout {repository} /opt/aster",
            f"cd /opt/aster && git fetch --depth=1 origin {commit}",
            f"cd /opt/aster && git checkout --detach {commit}",
            "cd /opt/aster && python -m pip install -e '.[cuda,liger,tracking,frontier]'",
        )
    )
    app = modal.App.lookup(
        plan["app_name"], environment_name=environment_name, create_if_missing=True
    )
    volumes = (
        {dataset_mount: dataset_volume} if dataset_volume is not None else {}
    )
    volumes.update({
            mount: modal.Volume.from_name(
                name,
                environment_name=environment_name,
                create_if_missing=True,
                version=int(plan["volume_version"]),
            )
            for mount, name in plan["volumes"].items()
            if mount != dataset_mount
        })
    secrets = [
        modal.Secret.from_name(name, environment_name=environment_name)
        for name in plan["secrets"]
    ]
    environment = {
        "ASTERLM_REMOTE_PROVIDER": "modal",
        "ASTERLM_HUB_RESUME_ROOT": "/var/cache/aster/hub-resume",
        "ASTERLM_REMOTE_RUN_ROOT": "/opt/aster/runs",
        "HF_HOME": "/var/cache/aster/huggingface",
        "TRITON_CACHE_DIR": "/var/cache/aster/triton",
        "TORCH_EXTENSIONS_DIR": "/var/cache/aster/torch_extensions",
        "XDG_CACHE_HOME": "/var/cache/aster/xdg",
        "WANDB_DIR": "/opt/aster/runs/wandb",
    }
    environment = cuda_allocator_environment(environment)
    if job_kind == "training":
        environment["ASTERLM_REMOTE_CONTRACT_B64"] = base64.b64encode(payload_bytes).decode("ascii")
        entrypoint = "scripts/cloud/modal_entrypoint.py"
    else:
        environment["ASTERLM_QUALIFICATION_SPEC_B64"] = base64.b64encode(payload_bytes).decode("ascii")
        environment["ASTERLM_QUALIFICATION_ID"] = plan["contract_id"]
        entrypoint = "scripts/cloud/modal_qualification_entrypoint.py"
    failures: list[dict[str, str]] = []
    for attempt in plan["attempts"]:
        if attempt.get("blockers"):
            continue
        try:
            sandbox = modal.Sandbox.create(
                "python",
                entrypoint,
                app=app,
                tags={
                    "aster_contract": plan["contract_id"],
                    "aster_profile": plan["profile_alias"],
                    "aster_gpu": attempt["gpu"],
                    "aster_job_kind": job_kind,
                },
                image=image,
                env=environment,
                secrets=secrets,
                volumes=volumes,
                timeout=int(plan["timeout_seconds"]),
                workdir="/opt/aster",
                gpu=attempt["gpu"],
                environment_name=environment_name,
            )
        except ModalError as exc:
            failures.append({"gpu": attempt["gpu"], "error": str(exc)[-2000:]})
            continue
        result = {
            "status": "dispatched",
            "provider": "modal",
            "contract_id": plan["contract_id"],
            "profile_alias": plan["profile_alias"],
            "gpu": attempt["gpu"],
            "sandbox_id": sandbox.object_id,
            "dataset_cache_verification": cache_verification,
            "failures_before_success": failures,
        }
        print("ASTER_MODAL_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
        return
    raise RuntimeError("All Modal GPU capacity attempts failed: " + json.dumps(failures))


if __name__ == "__main__":
    main()
