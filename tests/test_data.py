from pathlib import Path

import pytest

from asterlm.config import SourceConfig
from asterlm.data.mixture import _fim_transform, iter_source


def test_missing_jsonl_is_treated_as_local(tmp_path: Path):
    source = SourceConfig(path=str(tmp_path / "missing.jsonl"), weight=1.0)
    with pytest.raises(FileNotFoundError):
        next(iter(iter_source(source, seed=1, shuffle_buffer=1)))


def test_fim_transform_preserves_all_content():
    import random

    text = "abcdefghijklmnopqrstuvwxyz" * 4
    transformed = _fim_transform(text, random.Random(7))
    prefix, rest = transformed.removeprefix("<|fim_prefix|>").split("<|fim_suffix|>", 1)
    suffix, middle = rest.split("<|fim_middle|>", 1)
    assert prefix + middle + suffix == text


def test_source_config_validates_fim_rate():
    with pytest.raises(ValueError, match="fim_rate"):
        SourceConfig(path="example", weight=1.0, fim_rate=1.01)


def test_packed_dataset_preserves_boundary_token(monkeypatch):
    import asterlm.data.packing as packing
    from asterlm.config import DataConfig, SourceConfig

    class FakeTokenizer:
        def token_to_id(self, token: str) -> int:
            return 99

        def encode(self, text: str) -> list[int]:
            return [ord(char) - 96 for char in text]

    class FakeTextMixture:
        def __init__(self, config, validation=False):
            pass

        def __iter__(self):
            yield "abcdefghi"

    monkeypatch.setattr(packing, "TextMixture", FakeTextMixture)
    config = DataConfig(
        sources=[SourceConfig(path="unused", weight=1.0)],
        add_eos_between_documents=False,
    )
    iterator = iter(packing.PackedTokenDataset(FakeTokenizer(), config, sequence_length=4))
    first = next(iterator)
    second = next(iterator)
    assert first["input_ids"].tolist() == [1, 2, 3, 4]
    assert first["labels"].tolist() == [2, 3, 4, 5]
    assert second["input_ids"].tolist() == [5, 6, 7, 8]
    assert second["labels"].tolist() == [6, 7, 8, 9]


def test_normalize_sharegpt_messages():
    from asterlm.data.tokenizer import normalize_messages

    messages = normalize_messages([
        {"from": "human", "value": "Question"},
        {"from": "gpt", "value": "Answer"},
    ])
    assert messages == [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "Answer"},
    ]


def test_message_format_includes_top_level_system():
    from asterlm.data.mixture import record_to_text

    source = SourceConfig(path="unused", weight=1.0, format="messages")
    text = record_to_text(
        {
            "system": "Be precise.",
            "messages": [{"from": "human", "value": "Question"}, {"from": "gpt", "value": "Answer"}],
        },
        source,
    )
    assert text.startswith("<|system|>\nBe precise.<|end|>")
    assert "<|assistant|>\nAnswer<|end|>" in text


def test_packed_dataset_masks_cross_document_transition(monkeypatch):
    import asterlm.data.packing as packing
    from asterlm.config import DataConfig, SourceConfig

    class FakeTokenizer:
        def token_to_id(self, token: str) -> int:
            return 99

        def encode(self, text: str) -> list[int]:
            return [ord(char) - 96 for char in text]

    class FakeTextMixture:
        def __init__(self, config, validation=False):
            pass

        def __iter__(self):
            yield "abcd"
            yield "efgh"

    monkeypatch.setattr(packing, "TextMixture", FakeTextMixture)
    config = DataConfig(sources=[SourceConfig(path="unused", weight=1.0)])
    iterator = iter(packing.PackedTokenDataset(FakeTokenizer(), config, sequence_length=4))
    first = next(iterator)
    second = next(iterator)
    assert first["labels"].tolist() == [2, 3, 4, 99]
    assert second["input_ids"].tolist()[0] == 99
    assert second["labels"].tolist()[0] == -100


def test_local_reader_ignores_control_and_cursor_files(tmp_path: Path):
    import json
    import pickle

    root = tmp_path / "corpus"
    root.mkdir()
    (root / "part-00000.jsonl").write_text(json.dumps({"text": "usable"}) + "\n", encoding="utf-8")
    (root / "state.json").write_text(json.dumps({"text": "must not train"}), encoding="utf-8")
    (root / "manifest.json").write_text(json.dumps({"text": "must not train"}), encoding="utf-8")
    (root / "cursor-00000001.pkl").write_bytes(pickle.dumps({"text": "binary state"}))

    source = SourceConfig(path=str(root), weight=1.0)
    records = list(iter_source(source, seed=1, shuffle_buffer=1))
    assert records == [{"text": "usable"}]


def test_missing_data_directory_is_treated_as_local(tmp_path: Path):
    source = SourceConfig(path="data/not-downloaded-yet", weight=1.0)
    with pytest.raises(FileNotFoundError, match="Configured local data source"):
        next(iter(iter_source(source, seed=1, shuffle_buffer=1)))


def test_local_records_are_sharded_without_duplication(monkeypatch, tmp_path: Path):
    import json
    from types import SimpleNamespace

    import asterlm.data.mixture as mixture

    root = tmp_path / "single-shard"
    root.mkdir()
    rows = [{"text": f"row-{index}"} for index in range(8)]
    (root / "part.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    source = SourceConfig(path=str(root), weight=1.0)

    monkeypatch.setattr(mixture, "get_worker_info", lambda: SimpleNamespace(id=0, num_workers=2))
    first = list(mixture.iter_source(source, seed=1, shuffle_buffer=1))
    monkeypatch.setattr(mixture, "get_worker_info", lambda: SimpleNamespace(id=1, num_workers=2))
    second = list(mixture.iter_source(source, seed=1, shuffle_buffer=1))

    assert {row["text"] for row in first}.isdisjoint({row["text"] for row in second})
    assert sorted(first + second, key=lambda row: row["text"]) == sorted(rows, key=lambda row: row["text"])


def test_validation_routing_is_deterministic_and_monotonic():
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "clean_corpus.py"
    spec = importlib.util.spec_from_file_location("clean_corpus_for_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    digest = bytes.fromhex("1000000000000000" + "00" * 24)
    assert module.is_validation_digest(digest, 0.10)
    assert module.is_validation_digest(digest, 0.20)
    assert not module.is_validation_digest(digest, 0.01)
