from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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

        metadata = {
            "reason": reason,
            "step": step,
            "tokens_seen": tokens_seen,
            "checkpoint": checkpoint.name,
            "uploaded_at_unix": started,
            "include_optimizer": self.include_optimizer,
        }
        state_path = root / "hub_sync_state.json"
        state_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        for path in (
            root / "run_manifest.json",
            root / "metrics.jsonl",
            root / "latest.txt",
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

        ignore_patterns = None if self.include_optimizer else ["trainer_state.pt"]
        self.api.upload_folder(
            folder_path=str(checkpoint),
            path_in_repo=f"{prefix}/checkpoints/{checkpoint.name}",
            repo_id=self.repo_id,
            repo_type="model",
            revision=self.revision,
            ignore_patterns=ignore_patterns,
            commit_message=f"Sync {root.name}: {reason} at {tokens_seen:,} tokens",
        )
        metadata["seconds"] = time.time() - started
        metadata["status"] = "complete"
        state_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return metadata
