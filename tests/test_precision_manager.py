from types import SimpleNamespace

import torch

from asterlm.training.precision import PrecisionManager


def _config():
    return SimpleNamespace(
        precision_backend="amp",
        activation_offload=False,
        activation_offload_pin_memory=True,
    )


def test_precision_manager_slots_allow_internal_runtime_state():
    manager = PrecisionManager(
        config=_config(),
        device=torch.device("cpu"),
        autocast_dtype=torch.float32,
    )
    assert manager._te is None
    assert manager._recipe is None


def test_precision_manager_amp_forward_context_constructs():
    manager = PrecisionManager(
        config=_config(),
        device=torch.device("cpu"),
        autocast_dtype=torch.float32,
    )
    with manager.forward_context():
        value = torch.tensor(1.0)
    assert value.item() == 1.0
