"""AsterLM process-wide Python startup controls.

Loaded automatically by Python when scripts/python_startup is on PYTHONPATH.
The safe download supervisor enables ASTERLM_FORCE_IPV4 by default because some
networks advertise IPv6 but cannot complete outbound TLS connections reliably.
"""
from __future__ import annotations

import os
import socket
from typing import Any


def _enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


if _enabled("ASTERLM_FORCE_IPV4") and not getattr(socket, "_asterlm_ipv4_patched", False):
    _original_getaddrinfo = socket.getaddrinfo

    def _asterlm_getaddrinfo(
        host: Any,
        port: Any,
        family: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> list[tuple[Any, ...]]:
        results = _original_getaddrinfo(host, port, family, type, proto, flags)
        if family in (0, socket.AF_UNSPEC):
            ipv4 = [result for result in results if result[0] == socket.AF_INET]
            if ipv4:
                return ipv4
        return results

    socket.getaddrinfo = _asterlm_getaddrinfo  # type: ignore[assignment]
    socket._asterlm_ipv4_patched = True  # type: ignore[attr-defined]
