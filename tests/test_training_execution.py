from __future__ import annotations

import pytest
import torch
from torch import nn

from asterlm import AsterConfig, TrainConfig
from asterlm.training.execution import resolve_execution_engine


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
