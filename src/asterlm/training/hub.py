from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from asterlm.artifacts import atomic_write_json, sha256_file


def _git_blob_sha1(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Return the object id used by Git for a non-LFS file."""

    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _lfs_sha256(repo_file: Any) -> str | None:
    lfs = getattr(repo_file, "lfs", None)
    if lfs is None:
        return None
    if isinstance(lfs, dict):
        value = lfs.get("sha256")
    else:
        value = getattr(lfs, "sha256", None)
    return str(value) if value else None


@dataclass(slots=True)
class HubRunSync:
    """Synchronous, resumable Hugging Face backup for experiment artifacts.

    Full uploads are intentionally tied to permanent milestones/final checkpoints by
    the trainer. Hugging Face's Xet-backed upload_folder is resumable and deduplicates
    chunks, so rerunning after a failed transfer does not resend committed content.
    """

    repo_id: str
    private: bool = True
    revision: str = "main"
    include_optimizer: bool = True
    storage_guard_bytes: int | None = None
    storage_hard_cap_bytes: int | None = None
    api: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        from huggingface_hub import HfApi

        self.api = HfApi()
        self.api.create_repo(
            repo_id=self.repo_id,
            repo_type="model",
            private=self.private,
            exist_ok=True,
        )

    def remote_logical_bytes(self) -> int:
        """Return the repository's logical file size, including LFS/Xet files."""

        entries = self.api.list_repo_tree(
            repo_id=self.repo_id,
            repo_type="model",
            revision=self.revision,
            recursive=True,
            expand=True,
        )
        return sum(int(getattr(entry, "size", 0) or 0) for entry in entries)

    @staticmethod
    def _planned_upload_bytes(root: Path, checkpoint: Path) -> int:
        """Pessimistic upload forecast; replacements are deliberately counted again."""

        files: set[Path] = {path for path in checkpoint.rglob("*") if path.is_file()}
        for path in (
            root / "run_manifest.json",
            root / "analysis_manifest.json",
            root / "experiment.json",
            root / "metrics.jsonl",
            root / "hub_sync_state.json",
        ):
            if path.is_file():
                files.add(path)
        for artifact_dir in (root / "tensorboard", root / "diagnostics"):
            if artifact_dir.is_dir():
                files.update(path for path in artifact_dir.rglob("*") if path.is_file())
        return sum(path.stat().st_size for path in files)

    def storage_preflight(self, *, root: Path, checkpoint: Path) -> dict[str, Any]:
        remote_bytes = self.remote_logical_bytes()
        planned_bytes = self._planned_upload_bytes(root, checkpoint)
        projected_bytes = remote_bytes + planned_bytes
        if self.storage_hard_cap_bytes is not None and projected_bytes > self.storage_hard_cap_bytes:
            raise RuntimeError(
                "Hugging Face storage hard cap would be exceeded: "
                f"remote={remote_bytes:,}, planned={planned_bytes:,}, "
                f"projected={projected_bytes:,}, cap={self.storage_hard_cap_bytes:,} bytes"
            )
        if self.storage_guard_bytes is not None and projected_bytes > self.storage_guard_bytes:
            raise RuntimeError(
                "Hugging Face operational storage guard would be exceeded: "
                f"remote={remote_bytes:,}, planned={planned_bytes:,}, "
                f"projected={projected_bytes:,}, guard={self.storage_guard_bytes:,} bytes. "
                "Raise the explicit guard only after reviewing retention."
            )
        return {
            "remote_logical_bytes_before": remote_bytes,
            "planned_upload_bytes_pessimistic": planned_bytes,
            "projected_logical_bytes_pessimistic": projected_bytes,
            "storage_guard_bytes": self.storage_guard_bytes,
            "storage_hard_cap_bytes": self.storage_hard_cap_bytes,
        }

    @staticmethod
    def _run_prefix(output_dir: Path) -> str:
        return f"runs/{output_dir.name}"

    def sync(
        self,
        *,
        output_dir: str | Path,
        checkpoint: str | Path,
        reason: str,
        step: int,
        tokens_seen: int,
    ) -> dict[str, Any]:
        root = Path(output_dir)
        checkpoint = Path(checkpoint)
        prefix = self._run_prefix(root)
        started = time.time()

        if not self.include_optimizer:
            raise RuntimeError(
                "A recoverable Hub checkpoint must include trainer_state.pt; "
                "use a separate model export for weights-only publication"
            )

        from asterlm.training.checkpoint import verify_checkpoint

        local_manifest = verify_checkpoint(checkpoint)
        metadata = {
            "reason": reason,
            "step": step,
            "tokens_seen": tokens_seen,
            "checkpoint": checkpoint.name,
            "uploaded_at_unix": started,
            "include_optimizer": self.include_optimizer,
            "checkpoint_manifest_sha256": sha256_file(
                checkpoint / "checkpoint_manifest.json"
            ),
        }
        state_path = root / "hub_sync_state.json"
        atomic_write_json(state_path, metadata)
        metadata["storage_preflight"] = self.storage_preflight(
            root=root,
            checkpoint=checkpoint,
        )
        atomic_write_json(state_path, metadata)

        for path in (
            root / "run_manifest.json",
            root / "analysis_manifest.json",
            root / "experiment.json",
            root / "metrics.jsonl",
            state_path,
        ):
            if not path.exists():
                continue
            self.api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=f"{prefix}/{path.name}",
                repo_id=self.repo_id,
                repo_type="model",
                revision=self.revision,
                commit_message=f"Sync {root.name}: {reason} metadata",
            )

        for artifact_dir in (root / "tensorboard", root / "diagnostics"):
            if artifact_dir.exists():
                self.api.upload_folder(
                    folder_path=str(artifact_dir),
                    path_in_repo=f"{prefix}/artifacts/{artifact_dir.name}",
                    repo_id=self.repo_id,
                    repo_type="model",
                    revision=self.revision,
                    commit_message=f"Sync {root.name}: {artifact_dir.name} at {tokens_seen:,} tokens",
                )

        commit_info = self.api.upload_folder(
            folder_path=str(checkpoint),
            path_in_repo=f"{prefix}/checkpoints/{checkpoint.name}",
            repo_id=self.repo_id,
            repo_type="model",
            revision=self.revision,
            commit_message=f"Sync {root.name}: {reason} at {tokens_seen:,} tokens",
        )
        verification = self.verify_remote_checkpoint(
            checkpoint=checkpoint,
            path_in_repo=f"{prefix}/checkpoints/{checkpoint.name}",
        )
        # Publish the movable pointer only after every checkpoint artifact has an
        # authoritative remote hash. An interrupted upload can never advertise a
        # missing/incomplete recovery point.
        latest_path = root / "latest.txt"
        if latest_path.is_file():
            self.api.upload_file(
                path_or_fileobj=str(latest_path),
                path_in_repo=f"{prefix}/latest.txt",
                repo_id=self.repo_id,
                repo_type="model",
                revision=self.revision,
                commit_message=f"Advance {root.name} latest to verified {checkpoint.name}",
            )
        metadata["seconds"] = time.time() - started
        metadata["status"] = "verified"
        metadata["verified_file_count"] = verification["verified_file_count"]
        metadata["remote_commit"] = getattr(commit_info, "oid", None)
        metadata["resume_state"] = local_manifest.get("resume_state", {})
        atomic_write_json(state_path, metadata)
        verification_dir = root / "hub-verifications"
        atomic_write_json(
            verification_dir / f"{checkpoint.name}.json",
            {**metadata, **verification},
        )
        return metadata

    def verify_remote_checkpoint(
        self,
        *,
        checkpoint: str | Path,
        path_in_repo: str,
    ) -> dict[str, Any]:
        """Fail closed unless every uploaded file has an authoritative remote hash."""

        checkpoint = Path(checkpoint)
        local_files = sorted(path for path in checkpoint.rglob("*") if path.is_file())
        remote_paths = [
            f"{path_in_repo}/{path.relative_to(checkpoint).as_posix()}"
            for path in local_files
        ]
        infos = self.api.get_paths_info(
            repo_id=self.repo_id,
            paths=remote_paths,
            expand=True,
            revision=self.revision,
            repo_type="model",
        )
        by_path = {str(info.path): info for info in infos}
        verified: list[dict[str, Any]] = []
        for local, remote_path in zip(local_files, remote_paths, strict=True):
            info = by_path.get(remote_path)
            if info is None:
                raise RuntimeError(f"Hub verification missing remote file: {remote_path}")
            local_size = local.stat().st_size
            if int(info.size) != local_size:
                raise RuntimeError(f"Hub verification size mismatch: {remote_path}")
            lfs_sha = _lfs_sha256(info)
            if lfs_sha is not None:
                local_hash = sha256_file(local)
                remote_hash = lfs_sha
                algorithm = "sha256"
            else:
                local_hash = _git_blob_sha1(local)
                remote_hash = str(getattr(info, "blob_id", ""))
                algorithm = "git-sha1"
            if not remote_hash or local_hash != remote_hash:
                raise RuntimeError(
                    f"Hub verification {algorithm} mismatch: {remote_path}"
                )
            verified.append(
                {
                    "path": remote_path,
                    "size_bytes": local_size,
                    "algorithm": algorithm,
                    "digest": local_hash,
                }
            )
        return {
            "status": "verified",
            "verified_file_count": len(verified),
            "files": verified,
        }
