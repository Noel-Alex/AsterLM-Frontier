from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from studio.stateful_data import StatefulLocalPackedDataset


class FakeTokenizer:
    vocab_size = 512

    def token_to_id(self, token: str) -> int:
        assert token == "<|endoftext|>"
        return 1

    def encode(self, text: str) -> list[int]:
        return [2 + (ord(char) % 251) for char in text]


def source(path: Path, weight: float, fim_rate: float = 0.0):
    return SimpleNamespace(
        path=str(path),
        text_field="text",
        weight=weight,
        fim_rate=fim_rate,
        format="text",
        messages_field="messages",
        prompt_field="prompt",
        response_field="response",
    )


def config(a: Path, b: Path):
    return SimpleNamespace(
        seed=1337,
        sources=[source(a, 0.7), source(b, 0.3)],
        validation_sources=[],
        add_eos_between_documents=True,
        mask_cross_document_loss=True,
        min_chars=1,
        max_chars=100000,
        quality_filters=False,
    )


def write_rows(path: Path, prefix: str, rows: int, payload_chars: int = 16) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with (path / "part-000.jsonl").open("w", encoding="utf-8") as handle:
        for index in range(rows):
            text = f"{prefix}-{index:03d}-" + (prefix[:1] * payload_chars)
            handle.write(json.dumps({"text": text}) + "\n")


def tensor_signature(batch):
    return (
        tuple(batch["input_ids"].tolist()),
        tuple(batch["labels"].tolist()),
    )


def test_fast_resume_is_exact_at_packed_token_boundary(tmp_path: Path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    # Long records intentionally leave multiple already-packed samples in the
    # residual buffer at checkpoint time. This catches generator-program-counter
    # bugs that short-record tests miss.
    write_rows(a, "alpha", 50, payload_chars=180)
    write_rows(b, "beta", 30, payload_chars=130)

    dataset = StatefulLocalPackedDataset(
        FakeTokenizer(),
        config(a, b),
        16,
    )
    iterator = iter(dataset)
    for _ in range(31):
        next(iterator)

    saved = dataset.state_dict()
    assert len(saved["buffer"]) >= 16
    expected = [tensor_signature(next(iterator)) for _ in range(25)]

    resumed = StatefulLocalPackedDataset(
        FakeTokenizer(),
        config(a, b),
        16,
    )
    resumed.load_state_dict(saved)
    resumed_iterator = iter(resumed)
    actual = [tensor_signature(next(resumed_iterator)) for _ in range(25)]

    assert actual == expected


def test_data_signature_rejects_modified_local_shard(tmp_path: Path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    write_rows(a, "alpha", 10)
    write_rows(b, "beta", 10)

    dataset = StatefulLocalPackedDataset(FakeTokenizer(), config(a, b), 16)
    iterator = iter(dataset)
    next(iterator)
    saved = dataset.state_dict()

    with (a / "part-000.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"text": "mutated"}) + "\n")

    resumed = StatefulLocalPackedDataset(FakeTokenizer(), config(a, b), 16)
    with pytest.raises(RuntimeError, match="signature"):
        resumed.load_state_dict(saved)
