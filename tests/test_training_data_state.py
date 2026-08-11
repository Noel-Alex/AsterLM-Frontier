from __future__ import annotations

from types import SimpleNamespace

import pytest

from asterlm.training.engine import Trainer


class FakeDataset:
    def __init__(self, state: dict | None = None) -> None:
        self.state = state or {"cursor": 0}
        self.loaded: dict | None = None

    def state_dict(self) -> dict:
        return self.state

    def load_state_dict(self, state: dict) -> None:
        self.loaded = state


def _trainer(step: int = 4, tokens: int = 128) -> Trainer:
    trainer = Trainer.__new__(Trainer)
    trainer.step = step
    trainer.tokens_seen = tokens
    trainer.train_loader = SimpleNamespace(dataset=FakeDataset({"cursor": 17}))
    trainer.validation_loader = SimpleNamespace(dataset=FakeDataset({"cursor": 5}))
    trainer.validation_iterator = object()
    return trainer


def test_trainer_data_state_carries_step_tokens_and_both_cursors():
    trainer = _trainer()
    assert trainer._training_data_state() == {
        "schema_version": 1,
        "step": 4,
        "tokens_seen": 128,
        "train": {"cursor": 17},
        "validation": {"cursor": 5},
    }


def test_trainer_restores_data_state_without_batch_replay():
    trainer = _trainer()
    state = trainer._training_data_state()
    restored = _trainer()
    restored.train_loader.dataset = FakeDataset()
    restored.validation_loader.dataset = FakeDataset()
    restored._restore_training_data_state(state)
    assert restored.train_loader.dataset.loaded == {"cursor": 17}
    assert restored.validation_loader.dataset.loaded == {"cursor": 5}


def test_trainer_rejects_cursor_identity_mismatch():
    trainer = _trainer()
    with pytest.raises(RuntimeError, match="step/token identity"):
        trainer._restore_training_data_state(
            {
                "schema_version": 1,
                "step": 3,
                "tokens_seen": 128,
                "train": {"cursor": 17},
                "validation": None,
            }
        )
