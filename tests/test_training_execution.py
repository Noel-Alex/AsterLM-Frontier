from __future__ import annotations

import json
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
    assert engine.plan.moe_implementation == "reference"
    assert engine.plan.moe_selection_source == "not_applicable_dense_ffn"
    assert engine.prepare_model(nn.Linear(4, 4)).__class__ is nn.Linear


def test_moe_backend_is_first_class_and_environment_conflicts_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = AsterConfig(ffn_type="moe")
    configured = TrainConfig(
        device="cpu", execution_autotune=False, moe_implementation="reference"
    )
    engine = resolve_execution_engine(model, configured, torch.device("cpu"))
    assert engine.plan.moe_implementation == "reference"
    assert engine.plan.moe_selection_source == "train_config"

    monkeypatch.setenv("ASTER_MOE_IMPL", "cutlass")
    with pytest.raises(RuntimeError, match="Conflicting MoE execution selections"):
        resolve_execution_engine(model, configured, torch.device("cpu"))


def test_verified_autotune_cache_is_consumed_and_manifested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = AsterConfig(ffn_type="moe")
    train = TrainConfig(device="cpu", execution_autotune=True)
    first = resolve_execution_engine(model, train, torch.device("cpu"))
    assert first.plan.autotune_cache_key
    cache = tmp_path / "autotune.json"
    cache.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "entries": {
                    first.plan.autotune_cache_key: {
                        "winner": "reference",
                        "status": "promoted",
                        "numerical_parity": True,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ASTERLM_EXECUTION_AUTOTUNE_CACHE", str(cache))
    resolved = resolve_execution_engine(model, train, torch.device("cpu"))
    assert resolved.plan.moe_implementation == "reference"
    assert resolved.plan.moe_selection_source == "verified_autotune_cache"


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


def test_git_commit_treats_unreadable_checkout_as_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = tmp_path / "Megatron-LM"
    monkeypatch.setattr(
        Path,
        "exists",
        lambda self: (_ for _ in ()).throw(PermissionError("denied")),
    )

    assert execution_module._git_commit(checkout) is None


def test_backend_probe_marks_local_engine_unsupported_for_distributed_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORLD_SIZE", "2")
    capabilities = probe_execution_backends(torch.device("cpu"), lock_path="missing.yaml")
    local = capabilities["aster_local"]
    assert not local.topology_supported
    assert not local.usable
