from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from asterlm import AsterConfig, TrainConfig
from asterlm.training import execution as execution_module
from asterlm.training.execution import probe_execution_backends, resolve_execution_engine


def test_local_execution_plan_is_explicit_and_auditable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    cfg = TrainConfig(device="cpu", compile=False, execution_backend="auto")
    engine = resolve_execution_engine(AsterConfig(), cfg, torch.device("cpu"))

    assert engine.plan.engine == "aster_local"
    assert engine.plan.topology.world_size == 1
    assert engine.plan.distributed_strategy == "none"
    assert not engine.plan.compile_enabled
    assert engine.prepare_model(nn.Linear(4, 4)).__class__ is nn.Linear


def test_unpromoted_external_engine_is_never_silently_claimed() -> None:
    cfg = TrainConfig(device="cpu", execution_backend="megatron_core")
    with pytest.raises(NotImplementedError, match="no promoted Aster adapter"):
        resolve_execution_engine(AsterConfig(), cfg, torch.device("cpu"))


def test_local_engine_rejects_accidental_distributed_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORLD_SIZE", "2")
    cfg = TrainConfig(device="cpu", execution_backend="aster_local")
    with pytest.raises(RuntimeError, match="single-process"):
        resolve_execution_engine(AsterConfig(), cfg, torch.device("cpu"))


def test_cuda_graph_training_must_pass_promotion_gate() -> None:
    cfg = TrainConfig(device="cpu", cuda_graphs=True)
    with pytest.raises(NotImplementedError, match="not promoted"):
        resolve_execution_engine(AsterConfig(), cfg, torch.device("cpu"))


def test_backend_probe_separates_source_package_adapter_and_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commit = "a" * 40
    lock = tmp_path / "sources.yaml"
    lock.write_text(
        "\n".join(
            [
                "schema_version: 1",
                f"download_root: {tmp_path.as_posix()}",
                "sources:",
                "  - id: megatron_core",
                "    repository: https://github.com/NVIDIA/Megatron-LM.git",
                f"    commit: {commit}",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "Megatron-LM").mkdir()
    monkeypatch.setattr(
        execution_module,
        "_git_commit",
        lambda path: commit if path.name == "Megatron-LM" else None,
    )
    monkeypatch.setattr(execution_module, "_find_module", lambda module: module == "megatron.core")
    monkeypatch.setattr(execution_module, "_distribution_version", lambda distribution: "test")

    capabilities = probe_execution_backends(torch.device("cpu"), lock_path=lock)

    local = capabilities["aster_local"]
    assert local.usable
    assert local.promoted
    assert local.adapter_implemented

    megatron = capabilities["megatron_core"]
    assert megatron.importable
    assert megatron.installed_version == "test"
    assert megatron.source_matches_lock
    assert megatron.observed_commit == commit
    assert not megatron.adapter_implemented
    assert not megatron.promoted
    assert not megatron.usable
    assert "Aster adapter is not implemented" in megatron.blockers


def test_backend_probe_marks_local_engine_unsupported_for_distributed_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORLD_SIZE", "2")
    capabilities = probe_execution_backends(torch.device("cpu"), lock_path="missing.yaml")
    local = capabilities["aster_local"]
    assert not local.topology_supported
    assert not local.usable
