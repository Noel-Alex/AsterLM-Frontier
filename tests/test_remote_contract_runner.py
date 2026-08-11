from __future__ import annotations

import importlib.util
import signal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "aster_remote_contract_runner", ROOT / "scripts/cloud/run_contract.py"
)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_remote_runner_forwards_sigterm_to_training_process_group(monkeypatch):
    handlers = {}
    kills = []

    class FakeChild:
        pid = 4242

        @staticmethod
        def poll():
            return None

        @staticmethod
        def wait():
            handlers[signal.SIGTERM](signal.SIGTERM, None)
            return 143

    def fake_signal(signum, handler):
        previous = handlers.get(signum, f"old-{signum}")
        handlers[signum] = handler
        return previous

    monkeypatch.setattr(RUNNER.subprocess, "Popen", lambda *args, **kwargs: FakeChild())
    monkeypatch.setattr(RUNNER.signal, "signal", fake_signal)
    monkeypatch.setattr(RUNNER.os, "killpg", lambda pid, signum: kills.append((pid, signum)))

    assert RUNNER._run_training(["python", "trainer.py"]) == 143
    assert kills == [(4242, signal.SIGTERM)]
    assert handlers[signal.SIGINT] == f"old-{signal.SIGINT}"
    assert handlers[signal.SIGTERM] == f"old-{signal.SIGTERM}"
