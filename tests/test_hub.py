from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from asterlm.artifacts import sha256_file
from asterlm.training.engine import Trainer
from asterlm.training.hub import HubRunSync, HubUploadQueue, HubUploadTask, _git_blob_sha1


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


class _RecordingSync:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[dict[str, object]] = []

    def sync(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("upload failed")
        return {"status": "verified", "seconds": 0.1}


def test_async_upload_queue_drains_verified_tasks(tmp_path: Path) -> None:
    sync = _RecordingSync()
    upload_queue = HubUploadQueue(sync, max_pending=1)  # type: ignore[arg-type]
    checkpoint = tmp_path / "checkpoint-1"
    task = HubUploadTask(tmp_path, checkpoint, "periodic", 1, 100)
    upload_queue.enqueue(task)
    assert checkpoint.resolve() in upload_queue.pending_checkpoints()
    report = upload_queue.drain(close=True)
    assert report["errors"] == []
    assert report["results"][0]["status"] == "verified"
    assert upload_queue.pending_checkpoints() == set()
    assert sync.calls[0]["checkpoint"] == checkpoint


def test_async_upload_queue_surfaces_worker_errors(tmp_path: Path) -> None:
    upload_queue = HubUploadQueue(_RecordingSync(fail=True), max_pending=1)  # type: ignore[arg-type]
    checkpoint = tmp_path / "checkpoint-1"
    upload_queue.enqueue(HubUploadTask(tmp_path, checkpoint, "periodic", 1, 100))
    report = upload_queue.drain(close=True)
    assert report["results"] == []
    assert "upload failed" in report["errors"][0]["error"]


def test_trainer_close_flush_preserves_queue_for_protection_audit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeQueue:
        def drain(self, *, close: bool = False) -> dict[str, list[object]]:
            assert close
            return {"results": [], "errors": []}

        @staticmethod
        def pending_checkpoints() -> set[Path]:
            return {tmp_path / "checkpoint-pending"}

    protected_calls: list[set[Path]] = []
    fake_trainer = type("FakeTrainer", (), {})()
    fake_trainer.hub_upload_queue = FakeQueue()
    fake_trainer.train_config = type(
        "FakeTrainConfig",
        (),
        {
            "hub_fail_on_error": True,
            "output_dir": str(tmp_path),
            "keep_last_checkpoints": 2,
            "checkpoint_pyramid_levels": 1,
            "checkpoint_local_budget_gib": None,
        },
    )()
    fake_trainer._log = lambda _payload: None
    monkeypatch.setattr(
        "asterlm.training.engine.prune_rolling_checkpoints",
        lambda _output, *, protected, **_kwargs: protected_calls.append(protected),
    )

    Trainer._flush_hub_uploads(fake_trainer, close=True)

    assert fake_trainer.hub_upload_queue is None
    assert protected_calls == [{tmp_path / "checkpoint-pending"}]
