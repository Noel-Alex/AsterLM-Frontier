from __future__ import annotations

from typing import Any

import torch

from asterlm.training import execution
from studio import server


class _Capability:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)


def test_execution_backend_status_is_cached_and_secret_free(monkeypatch) -> None:
    calls = 0

    def probe(_device: torch.device, **_kwargs: Any) -> dict[str, _Capability]:
        nonlocal calls
        calls += 1
        return {
            "aster_local": _Capability(
                {
                    "backend": "aster_local",
                    "installed_version": "test",
                    "source_matches_lock": None,
                    "adapter_implemented": True,
                    "promoted": True,
                    "topology_supported": True,
                    "usable": True,
                    "blockers": [],
                }
            )
        }

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(execution, "probe_execution_backends", probe)
    monkeypatch.setitem(server._EXECUTION_BACKEND_CACHE, "rows", [])
    monkeypatch.setitem(server._EXECUTION_BACKEND_CACHE, "updated", 0.0)

    first = server.execution_backend_status(ttl_seconds=60)
    second = server.execution_backend_status(ttl_seconds=60)

    assert first == second
    assert first[0]["backend"] == "aster_local"
    assert first[0]["usable"] is True
    assert calls == 1
