from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from asterlm.artifacts import sha256_file
from asterlm.training.hub import HubRunSync, _git_blob_sha1


@dataclass
class _Lfs:
    sha256: str


@dataclass
class _RemoteFile:
    path: str
    size: int
    blob_id: str
    lfs: _Lfs | None = None


class _FakeApi:
    def __init__(self, files: dict[str, _RemoteFile]) -> None:
        self.files = files

    def get_paths_info(self, *, paths: list[str], **_: object) -> list[_RemoteFile]:
        return [self.files[path] for path in paths if path in self.files]

    def list_repo_tree(self, **_: object) -> list[_RemoteFile]:
        return list(self.files.values())


def _sync(api: _FakeApi) -> HubRunSync:
    sync = object.__new__(HubRunSync)
    sync.repo_id = "owner/private-checkpoints"
    sync.private = True
    sync.revision = "main"
    sync.include_optimizer = True
    sync.storage_guard_bytes = None
    sync.storage_hard_cap_bytes = None
    sync.api = api
    return sync


def test_remote_checkpoint_verification_supports_git_and_lfs_hashes(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-00000001"
    checkpoint.mkdir()
    small = checkpoint / "model_config.yaml"
    large = checkpoint / "model.safetensors"
    small.write_text("model: tiny\n", encoding="utf-8")
    large.write_bytes(b"weights" * 100)
    prefix = "runs/demo/checkpoints/checkpoint-00000001"
    api = _FakeApi(
        {
            f"{prefix}/model.safetensors": _RemoteFile(
                f"{prefix}/model.safetensors",
                large.stat().st_size,
                "lfs-pointer-object",
                _Lfs(sha256_file(large)),
            ),
            f"{prefix}/model_config.yaml": _RemoteFile(
                f"{prefix}/model_config.yaml",
                small.stat().st_size,
                _git_blob_sha1(small),
            ),
        }
    )

    result = _sync(api).verify_remote_checkpoint(
        checkpoint=checkpoint,
        path_in_repo=prefix,
    )

    assert result["status"] == "verified"
    assert result["verified_file_count"] == 2
    assert {item["algorithm"] for item in result["files"]} == {"git-sha1", "sha256"}


def test_remote_checkpoint_verification_fails_closed_on_hash_mismatch(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-00000001"
    checkpoint.mkdir()
    local = checkpoint / "trainer_state.pt"
    local.write_bytes(b"optimizer state")
    prefix = "runs/demo/checkpoints/checkpoint-00000001"
    remote_path = f"{prefix}/trainer_state.pt"
    api = _FakeApi(
        {
            remote_path: _RemoteFile(
                remote_path,
                local.stat().st_size,
                "lfs-pointer-object",
                _Lfs("0" * 64),
            )
        }
    )

    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        _sync(api).verify_remote_checkpoint(
            checkpoint=checkpoint,
            path_in_repo=prefix,
        )


def test_storage_preflight_uses_pessimistic_logical_size(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-00000001"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"x" * 40)
    api = _FakeApi({"old": _RemoteFile("old", 50, "blob")})
    sync = _sync(api)
    sync.storage_guard_bytes = 100
    sync.storage_hard_cap_bytes = 120

    result = sync.storage_preflight(root=tmp_path, checkpoint=checkpoint)

    assert result["remote_logical_bytes_before"] == 50
    assert result["planned_upload_bytes_pessimistic"] == 40
    assert result["projected_logical_bytes_pessimistic"] == 90


def test_storage_preflight_fails_before_operational_guard(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-00000001"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.pt").write_bytes(b"x" * 60)
    sync = _sync(_FakeApi({"old": _RemoteFile("old", 50, "blob")}))
    sync.storage_guard_bytes = 100
    sync.storage_hard_cap_bytes = 120

    with pytest.raises(RuntimeError, match="operational storage guard"):
        sync.storage_preflight(root=tmp_path, checkpoint=checkpoint)
