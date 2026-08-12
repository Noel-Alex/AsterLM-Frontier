from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from .gcp import GcpProfile
from .modal_cache import build_modal_cache_stage_plan


def build_gcp_cache_stage_plan(
    manifest_path: str | Path,
    profile: GcpProfile,
    *,
    root: str | Path,
    verify_local_hashes: bool = False,
) -> dict[str, Any]:
    """Build a clean-manifest-only Cloud Storage staging plan with no provider call."""

    local = build_modal_cache_stage_plan(
        manifest_path,
        root=root,
        profile_alias=profile.alias,
        modal_environment="not-applicable",
        volume_name="not-applicable",
        verify_local_hashes=verify_local_hashes,
    )
    files = [
        {
            **row,
            "uri": f"gs://{profile.bucket}/{profile.dataset_prefix}/{row['volume_path']}",
        }
        for row in local["files"]
    ]
    manifest = files[-1]
    return {
        "schema_version": 1,
        "provider": "gcp",
        "status": "ready",
        "profile_alias": profile.alias,
        "gcloud_configuration": profile.gcloud_configuration,
        "project_id": profile.project_id,
        "bucket": profile.bucket,
        "dataset_prefix": profile.dataset_prefix,
        "manifest": {
            "local_path": manifest["local_path"],
            "uri": manifest["uri"],
            "sha256": manifest["sha256"],
        },
        "files": files,
        "file_count": len(files),
        "total_bytes": local["total_bytes"],
        "raw_corpus_included": False,
        "local_hashes_verified": bool(verify_local_hashes),
        "upload_policy": (
            "describe object custom SHA-256 and size, upload only missing/stale sealed "
            "artifacts, publish the clean manifest last"
        ),
    }


def execute_gcp_cache_stage(plan: dict[str, Any]) -> dict[str, Any]:
    if plan.get("provider") != "gcp" or plan.get("status") != "ready":
        raise RuntimeError("Refusing a blocked or non-GCP cache-stage plan")
    if not plan.get("local_hashes_verified"):
        raise RuntimeError("Execute requires full local artifact hash verification")
    uploaded = 0
    reused = 0
    for row in plan["files"]:
        common = [
            "--configuration",
            str(plan["gcloud_configuration"]),
            "--project",
            str(plan["project_id"]),
        ]
        described = subprocess.run(
            [
                "gcloud",
                "storage",
                "objects",
                "describe",
                row["uri"],
                "--format=json(custom_fields,size)",
                *common,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        current: dict[str, Any] = {}
        if described.returncode == 0:
            try:
                current = json.loads(described.stdout)
            except json.JSONDecodeError:
                current = {}
        custom = current.get("custom_fields") or {}
        if (
            isinstance(custom, dict)
            and custom.get("aster-sha256") == row["sha256"]
            and int(current.get("size", -1)) == int(row["size_bytes"])
        ):
            reused += 1
            continue
        copied = subprocess.run(
            [
                "gcloud",
                "storage",
                "cp",
                row["local_path"],
                row["uri"],
                f"--custom-metadata=aster-sha256={row['sha256']}",
                *common,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if copied.returncode:
            raise RuntimeError(
                f"GCP cache upload failed for {row['volume_path']}: {copied.stderr[-2000:]}"
            )
        uploaded += 1
    return {
        "status": "staged" if uploaded else "already_current",
        "manifest_sha256": plan["manifest"]["sha256"],
        "uploaded_files": uploaded,
        "reused_files": reused,
        "logical_bytes": plan["total_bytes"],
    }
