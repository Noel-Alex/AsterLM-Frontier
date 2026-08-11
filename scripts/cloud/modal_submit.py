#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit one immutable Aster contract to Modal")
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if plan.get("provider") != "modal" or plan.get("blockers"):
        raise RuntimeError("Refusing a non-Modal or blocked launch plan")
    if os.environ.get("MODAL_PROFILE") != plan["profile_alias"]:
        raise RuntimeError("MODAL_PROFILE does not match the launch plan")

    contract_path = Path(plan["contract_path"])
    contract_bytes = contract_path.read_bytes()
    if hashlib.sha256(contract_bytes).hexdigest() != plan["contract_sha256"]:
        raise RuntimeError("Modal contract changed after launch-plan creation")

    # Import only after MODAL_PROFILE has been isolated in this subprocess.
    import modal
    from modal.exception import Error as ModalError

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
    environment_name = plan["modal_environment"]
    app = modal.App.lookup(
        plan["app_name"], environment_name=environment_name, create_if_missing=True
    )
    volumes = {
        mount: modal.Volume.from_name(
            name,
            environment_name=environment_name,
            create_if_missing=True,
            version=int(plan["volume_version"]),
        )
        for mount, name in plan["volumes"].items()
    }
    secrets = [
        modal.Secret.from_name(name, environment_name=environment_name)
        for name in plan["secrets"]
    ]
    environment = {
        "ASTERLM_REMOTE_PROVIDER": "modal",
        "ASTERLM_REMOTE_CONTRACT_B64": base64.b64encode(contract_bytes).decode("ascii"),
        "ASTERLM_HUB_RESUME_ROOT": "/var/cache/aster/hub-resume",
        "ASTERLM_REMOTE_RUN_ROOT": "/opt/aster/runs",
        "HF_HOME": "/var/cache/aster/huggingface",
        "WANDB_DIR": "/opt/aster/runs/wandb",
    }
    failures: list[dict[str, str]] = []
    for attempt in plan["attempts"]:
        if attempt.get("blockers"):
            continue
        try:
            sandbox = modal.Sandbox.create(
                "python",
                "scripts/cloud/modal_entrypoint.py",
                app=app,
                tags={
                    "aster_contract": plan["contract_id"],
                    "aster_profile": plan["profile_alias"],
                    "aster_gpu": attempt["gpu"],
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
            "failures_before_success": failures,
        }
        print("ASTER_MODAL_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
        return
    raise RuntimeError("All Modal GPU capacity attempts failed: " + json.dumps(failures))


if __name__ == "__main__":
    main()
