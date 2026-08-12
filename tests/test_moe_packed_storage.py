from __future__ import annotations

import torch
from torch import nn

from asterlm.layers.moe_grouped_cutlass import (
    _CachedParameterStack,
    materialize_parameter_storage,
    pack_parameter_storage,
)
from asterlm.optim.muon import Muon
from asterlm.training.checkpoint import checkpoint_compatible_parameter_storage


def test_packed_expert_storage_preserves_keys_values_and_independent_gradients():
    torch.manual_seed(17)
    experts = nn.ModuleList([nn.Linear(5, 7, bias=False) for _ in range(3)])
    parameters = [expert.weight for expert in experts]
    names_before = tuple(experts.state_dict())
    values_before = torch.stack([parameter.detach().clone() for parameter in parameters])

    packed = pack_parameter_storage(parameters)

    assert tuple(experts.state_dict()) == names_before
    torch.testing.assert_close(packed, values_before)
    storage_ptr = packed.untyped_storage().data_ptr()
    assert all(parameter.untyped_storage().data_ptr() == storage_ptr for parameter in parameters)
    assert [parameter.storage_offset() for parameter in parameters] == [0, 35, 70]

    proxy = _CachedParameterStack.apply(packed, *parameters)
    proxy.square().sum().backward()
    for index, parameter in enumerate(parameters):
        torch.testing.assert_close(parameter.grad, 2.0 * values_before[index])


def test_packed_expert_storage_tracks_optimizer_updates_without_restacking():
    torch.manual_seed(23)
    experts = nn.ModuleList([nn.Linear(5, 7, bias=False) for _ in range(3)])
    parameters = [expert.weight for expert in experts]
    packed = pack_parameter_storage(parameters)
    optimizer = torch.optim.AdamW(parameters, lr=1e-2, foreach=True)

    before = packed.detach().clone()
    proxy = _CachedParameterStack.apply(packed, *parameters)
    proxy.square().sum().backward()
    optimizer.step()

    assert not torch.equal(packed, before)
    torch.testing.assert_close(packed, torch.stack([p.detach() for p in parameters]))


def test_packed_expert_storage_tracks_megabatched_muon_updates():
    torch.manual_seed(29)
    parameters = [nn.Parameter(torch.randn(8, 16)) for _ in range(4)]
    packed = pack_parameter_storage(parameters)
    for parameter in parameters:
        parameter.grad = torch.randn_like(parameter)
    optimizer = Muon(parameters, lr=0.01, megabatch=True)

    optimizer.step()

    torch.testing.assert_close(
        packed,
        torch.stack([parameter.detach() for parameter in parameters]),
    )


def test_materialized_expert_storage_round_trips_through_safetensors(tmp_path):
    from safetensors.torch import load_model, save_model

    class PackedExperts(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.experts = nn.ModuleList(
                [nn.Linear(5, 7, bias=False) for _ in range(3)]
            )
            self.packed: torch.Tensor | None = None

        def pack_grouped_expert_storage(self) -> dict[str, float]:
            self.packed = pack_parameter_storage(
                [expert.weight for expert in self.experts]
            )
            return {"packed": 1.0}

        def materialize_grouped_expert_storage(self) -> dict[str, float]:
            if self.packed is None:
                return {}
            materialize_parameter_storage(
                [expert.weight for expert in self.experts]
            )
            self.packed = None
            return {"materialized": 1.0}

    source = PackedExperts()
    expected = {name: value.detach().clone() for name, value in source.state_dict().items()}
    source.pack_grouped_expert_storage()
    path = tmp_path / "packed.safetensors"
    with checkpoint_compatible_parameter_storage(source):
        save_model(source, str(path))
        storage_ptrs = {
            expert.weight.untyped_storage().data_ptr() for expert in source.experts
        }
        assert len(storage_ptrs) == len(source.experts)
    assert source.packed is not None

    target = PackedExperts()
    target.pack_grouped_expert_storage()
    with checkpoint_compatible_parameter_storage(target):
        missing, unexpected = load_model(target, str(path), strict=True)
    assert not missing and not unexpected
    assert target.packed is not None
    for name, value in target.state_dict().items():
        torch.testing.assert_close(value, expected[name])


def test_packed_expert_storage_rejects_mixed_shapes():
    parameters = [
        nn.Parameter(torch.zeros(2, 3)),
        nn.Parameter(torch.zeros(3, 2)),
    ]
    try:
        pack_parameter_storage(parameters)
    except ValueError as exc:
        assert "share shape" in str(exc)
    else:
        raise AssertionError("mixed expert shapes were packed")
