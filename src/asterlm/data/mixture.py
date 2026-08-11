from __future__ import annotations

import gzip
import io
import json
import random
import re
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

from torch.utils.data import get_worker_info

from asterlm.config import DataConfig, SourceConfig

from .tokenizer import format_chat, normalize_messages

_REPEATED_CHAR = re.compile(r"(.)\1{40,}")


def _nested_get(record: dict[str, Any], field: str) -> Any:
    value: Any = record
    for part in field.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _quality_ok(text: str, config: DataConfig) -> bool:
    length = len(text)
    if length < config.min_chars or length > config.max_chars:
        return False
    if not config.quality_filters:
        return True
    if "\x00" in text or _REPEATED_CHAR.search(text):
        return False
    visible = sum(not c.isspace() for c in text)
    alpha = sum(c.isalpha() for c in text)
    if visible == 0 or alpha / visible < 0.08:
        return False
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 8 and len(set(lines)) / len(lines) < 0.35:
        return False
    return True


def _fim_transform(text: str, rng: random.Random) -> str:
    """Apply prefix-suffix-middle infilling to a document at character boundaries."""
    if len(text) < 32:
        return text
    left = rng.randrange(1, len(text) - 1)
    right = rng.randrange(left, len(text))
    prefix, middle, suffix = text[:left], text[left:right], text[right:]
    return f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>{middle}"


def record_to_text(record: dict[str, Any], source: SourceConfig) -> str | None:
    if source.format == "messages":
        messages = _nested_get(record, source.messages_field)
        if messages is None:
            return None
        normalized = normalize_messages(messages)
        system = record.get("system")
        chat_kwargs = record.get("chat_template_kwargs")
        if system is None and isinstance(chat_kwargs, dict):
            system = chat_kwargs.get("system") or chat_kwargs.get("system_prompt")
        if system and (not normalized or normalized[0]["role"] != "system"):
            normalized.insert(0, {"role": "system", "content": str(system)})
        return format_chat(normalized)
    if source.format == "prompt_response":
        prompt = _nested_get(record, source.prompt_field)
        response = _nested_get(record, source.response_field)
        if prompt is None or response is None:
            return None
        return format_chat(
            [
                {"role": "user", "content": str(prompt)},
                {"role": "assistant", "content": str(response)},
            ]
        )
    value = _nested_get(record, source.text_field)
    if value is None:
        # Several reasoning datasets use a solution field but may change naming.
        for candidate in (
            "text",
            "content",
            "deepseek_solution",
            "generated_solution",
            "solution",
            "response",
        ):
            value = record.get(candidate)
            if value is not None:
                break
    return None if value is None else str(value)


def _is_local_data_file(path: Path) -> bool:
    name = path.name.lower()
    if name.startswith("."):
        return False
    if name in {
        "state.json",
        "manifest.json",
        "download_manifest.json",
        "shard_verification.json",
        "data_preflight.json",
        "cleaning_report.json",
        "prepare_summary.json",
        "clean_manifest.json",
    }:
        return False
    ignored_suffixes = (".partial", ".tmp", ".pkl", ".lock", ".log", ".sqlite", ".db")
    if name.startswith("cursor-") or name.endswith(ignored_suffixes):
        return False
    return name.endswith(
        (
            ".jsonl",
            ".jsonl.gz",
            ".jsonl.zst",
            ".json",
            ".txt",
            ".text",
            ".md",
            ".markdown",
            ".rst",
            ".py",
            ".c",
            ".cc",
            ".cpp",
            ".h",
            ".hpp",
            ".java",
            ".js",
            ".ts",
            ".tsx",
            ".jsx",
            ".rs",
            ".go",
            ".html",
            ".xml",
            ".tex",
        )
    )


def local_data_paths(path: str | Path) -> list[Path]:
    """Return only record-bearing local files, excluding pipeline control metadata."""

    root = Path(path)
    candidates = sorted(root.rglob("*")) if root.is_dir() else [root]
    return [item for item in candidates if item.is_file() and _is_local_data_file(item)]


def _iter_local_file(item: Path, source: SourceConfig) -> Iterator[dict[str, Any]]:
    name = item.name.lower()
    if name.endswith(".jsonl.gz"):
        handle_ctx = gzip.open(item, "rt", encoding="utf-8")
        json_mode = "jsonl"
    elif name.endswith(".jsonl.zst"):
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise ImportError("Install zstandard to read .jsonl.zst corpora") from exc
        raw = item.open("rb")
        stream = zstd.ZstdDecompressor().stream_reader(raw)
        handle_ctx = io.TextIOWrapper(stream, encoding="utf-8")
        json_mode = "jsonl"
    elif item.suffix.lower() in {".jsonl", ".json"}:
        handle_ctx = item.open("r", encoding="utf-8")
        json_mode = "json" if item.suffix.lower() == ".json" else "jsonl"
    else:
        text = item.read_text(encoding="utf-8", errors="replace")
        yield {source.text_field: text}
        return

    with handle_ctx as handle:
        if json_mode == "json":
            data = json.load(handle)
            if isinstance(data, list):
                yield from (row for row in data if isinstance(row, dict))
            elif isinstance(data, dict):
                yield data
        else:
            for line in handle:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    if isinstance(record, dict):
                        yield record


def _iter_local(source: SourceConfig) -> Iterator[dict[str, Any]]:
    path = Path(source.path)
    paths = local_data_paths(path)
    worker = get_worker_info()
    worker_id = 0 if worker is None else worker.id
    num_workers = 1 if worker is None else worker.num_workers
    record_index = 0
    for item in paths:
        for record in _iter_local_file(item, source):
            if record_index % num_workers == worker_id:
                yield record
            record_index += 1

def _iter_hf(source: SourceConfig, seed: int, shuffle_buffer: int) -> Iterator[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install `datasets` to stream Hugging Face corpora") from exc

    kwargs: dict[str, Any] = {
        "path": source.path,
        "name": source.name,
        "split": source.split,
        "streaming": source.streaming,
        "trust_remote_code": source.trust_remote_code,
    }
    if source.revision:
        kwargs["revision"] = source.revision
    if source.data_files:
        kwargs["data_files"] = source.data_files
    dataset = load_dataset(**kwargs)
    if source.streaming and shuffle_buffer > 1:
        dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)
    worker = get_worker_info()
    if worker is not None and hasattr(dataset, "shard"):
        dataset = dataset.shard(num_shards=worker.num_workers, index=worker.id)
    yield from dataset


def _looks_like_local_path(value: str) -> bool:
    path = Path(value)
    normalized = value.replace("\\", "/")
    return (
        path.is_absolute()
        or normalized.startswith(("./", "../", "data/", "artifacts/", "runs/"))
        or path.suffix.lower() in {".txt", ".md", ".json", ".jsonl", ".gz", ".zst", ".py"}
    )


def iter_source(source: SourceConfig, seed: int, shuffle_buffer: int) -> Iterator[dict[str, Any]]:
    path = Path(source.path)
    if path.exists():
        yield from _iter_local(source)
    elif _looks_like_local_path(source.path):
        raise FileNotFoundError(
            f"Configured local data source does not exist: {source.path}. "
            "Create it first or remove it from the mixture."
        )
    else:
        yield from _iter_hf(source, seed, shuffle_buffer)


class StatefulSourceIterator:
    """Checkpointable source cursor with recovery bounded by one local shard.

    Plain JSONL can eventually use a byte-offset index. Compressed shards are replayed
    only to the saved row inside the current bounded shard, so restore time is constant
    with respect to total training tokens rather than proportional to the whole run.
    Hugging Face iterable datasets use their native state_dict cursor.
    """

    def __init__(self, source: SourceConfig, seed: int, shuffle_buffer: int) -> None:
        if get_worker_info() is not None:
            raise RuntimeError("Checkpointable data iteration requires num_workers=0")
        self.source = source
        self.seed = seed
        self.shuffle_buffer = shuffle_buffer
        self.path = Path(source.path)
        self.kind = "local" if self.path.exists() else "huggingface"
        if self.kind == "huggingface" and _looks_like_local_path(source.path):
            raise FileNotFoundError(f"Configured local data source does not exist: {source.path}")
        self.paths = local_data_paths(self.path) if self.kind == "local" else []
        self.file_index = 0
        self.record_index = 0
        self._local_iterator: Iterator[dict[str, Any]] | None = None
        self._hf_dataset: Any | None = None
        self._hf_iterator: Iterator[dict[str, Any]] | None = None
        self._pending_hf_state: dict[str, Any] | None = None

    def __iter__(self) -> StatefulSourceIterator:
        return self

    def _local_layout(self) -> list[dict[str, Any]]:
        return [
            {
                "path": (
                    path.relative_to(self.path).as_posix()
                    if self.path.is_dir()
                    else path.name
                ),
                "size": path.stat().st_size,
            }
            for path in self.paths
        ]

    def _open_local(self) -> None:
        while self.file_index < len(self.paths):
            iterator = _iter_local_file(self.paths[self.file_index], self.source)
            try:
                for _ in range(self.record_index):
                    next(iterator)
            except StopIteration:
                self.file_index += 1
                self.record_index = 0
                continue
            self._local_iterator = iterator
            return
        raise StopIteration

    def _open_hf(self) -> None:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError("Install `datasets` to stream Hugging Face corpora") from exc
        kwargs: dict[str, Any] = {
            "path": self.source.path,
            "name": self.source.name,
            "split": self.source.split,
            "streaming": self.source.streaming,
            "trust_remote_code": self.source.trust_remote_code,
        }
        if self.source.revision:
            kwargs["revision"] = self.source.revision
        if self.source.data_files:
            kwargs["data_files"] = self.source.data_files
        dataset = load_dataset(**kwargs)
        if self.source.streaming and self.shuffle_buffer > 1:
            dataset = dataset.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer)
        if self._pending_hf_state is not None:
            load = getattr(dataset, "load_state_dict", None)
            if not callable(load):
                raise RuntimeError("Hugging Face source does not support exact cursor restore")
            load(deepcopy(self._pending_hf_state))
        self._hf_dataset = dataset
        self._hf_iterator = iter(dataset)

    def __next__(self) -> dict[str, Any]:
        if self.kind == "huggingface":
            if self._hf_iterator is None:
                self._open_hf()
            assert self._hf_iterator is not None
            return next(self._hf_iterator)
        while True:
            if self._local_iterator is None:
                self._open_local()
            assert self._local_iterator is not None
            try:
                record = next(self._local_iterator)
            except StopIteration:
                self.file_index += 1
                self.record_index = 0
                self._local_iterator = None
                continue
            self.record_index += 1
            return record

    def state_dict(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "schema_version": 1,
            "kind": self.kind,
            "source": asdict(self.source),
            "seed": self.seed,
            "shuffle_buffer": self.shuffle_buffer,
        }
        if self.kind == "local":
            state.update(
                {
                    "shards": self._local_layout(),
                    "file_index": self.file_index,
                    "record_index": self.record_index,
                }
            )
        else:
            cursor = self._pending_hf_state
            method = getattr(self._hf_dataset, "state_dict", None)
            if callable(method):
                cursor = method()
            if cursor is None and self._hf_dataset is not None:
                raise RuntimeError("Hugging Face source does not expose a checkpointable cursor")
            state["hf_cursor"] = deepcopy(cursor)
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("schema_version") != 1 or state.get("kind") != self.kind:
            raise ValueError("Source cursor schema/kind mismatch")
        if state.get("source") != asdict(self.source):
            raise ValueError("Source configuration changed since checkpoint")
        if int(state.get("seed")) != self.seed or int(state.get("shuffle_buffer")) != self.shuffle_buffer:
            raise ValueError("Source cursor seed or shuffle buffer mismatch")
        if self.kind == "local":
            if state.get("shards") != self._local_layout():
                raise ValueError("Local source shard list changed since checkpoint")
            self.file_index = int(state["file_index"])
            self.record_index = int(state["record_index"])
            self._local_iterator = None
        else:
            self._pending_hf_state = deepcopy(state.get("hf_cursor"))
            self._hf_dataset = None
            self._hf_iterator = None


class RecordMixtureIterator:
    def __init__(self, config: DataConfig, sources: list[SourceConfig]) -> None:
        self.config = config
        self.sources = sources
        self.rng = random.Random(config.seed)
        self.weights = [source.weight for source in sources]
        self.source_epochs = [0] * len(sources)
        self.iterators = [
            StatefulSourceIterator(source, config.seed + index, config.shuffle_buffer)
            for index, source in enumerate(sources)
        ]

    def __iter__(self) -> RecordMixtureIterator:
        return self

    def __next__(self) -> tuple[dict[str, Any], SourceConfig]:
        while True:
            index = self.rng.choices(range(len(self.sources)), weights=self.weights, k=1)[0]
            try:
                record = next(self.iterators[index])
            except StopIteration:
                self.source_epochs[index] += 1
                seed = self.config.seed + index + self.rng.randrange(1_000_000)
                self.iterators[index] = StatefulSourceIterator(
                    self.sources[index], seed, self.config.shuffle_buffer
                )
                record = next(self.iterators[index])
            if isinstance(record, dict):
                return record, self.sources[index]

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "rng": self.rng.getstate(),
            "source_epochs": list(self.source_epochs),
            "sources": [iterator.state_dict() for iterator in self.iterators],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("schema_version") != 1 or len(state.get("sources", [])) != len(self.sources):
            raise ValueError("Record mixture checkpoint does not match configured sources")
        self.rng.setstate(state["rng"])
        self.source_epochs = [int(value) for value in state["source_epochs"]]
        for iterator, cursor in zip(self.iterators, state["sources"], strict=True):
            iterator.load_state_dict(cursor)


class RecordMixture:
    def __init__(self, config: DataConfig, validation: bool = False) -> None:
        self.config = config
        self.sources = config.validation_sources if validation else config.sources
        if not self.sources:
            raise ValueError("No data sources configured")
        if any(source.weight <= 0 for source in self.sources):
            raise ValueError("All source weights must be positive")
        self._active: RecordMixtureIterator | None = None
        self._pending_state: dict[str, Any] | None = None

    def __iter__(self) -> Iterator[tuple[dict[str, Any], SourceConfig]]:
        if get_worker_info() is not None:
            raise RuntimeError("Checkpointable RecordMixture requires num_workers=0")
        iterator = RecordMixtureIterator(self.config, self.sources)
        if self._pending_state is not None:
            iterator.load_state_dict(self._pending_state)
            self._pending_state = None
        self._active = iterator
        return iterator

    def state_dict(self) -> dict[str, Any]:
        if self._active is not None:
            return self._active.state_dict()
        if self._pending_state is not None:
            return deepcopy(self._pending_state)
        return RecordMixtureIterator(self.config, self.sources).state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if self._active is not None:
            raise RuntimeError("Load mixture state before creating its iterator")
        self._pending_state = deepcopy(state)


class TextMixtureIterator:
    def __init__(self, parent: TextMixture, records: Iterator[tuple[dict[str, Any], SourceConfig]]) -> None:
        self.parent = parent
        self.records = records
        self.rng = random.Random(parent.config.seed + (1 if parent.validation else 0))

    def __iter__(self) -> TextMixtureIterator:
        return self

    def __next__(self) -> str:
        while True:
            record, source = next(self.records)
            text = record_to_text(record, source)
            if text is None:
                continue
            text = text.strip()
            if not _quality_ok(text, self.parent.config):
                continue
            if not self.parent.validation and source.fim_rate > 0 and self.rng.random() < source.fim_rate:
                text = _fim_transform(text, self.rng)
            return text

    def state_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, "rng": self.rng.getstate(), "records": self.parent.records.state_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("schema_version") != 1:
            raise ValueError("Text mixture cursor schema mismatch")
        self.rng.setstate(state["rng"])


class TextMixture:
    def __init__(self, config: DataConfig, validation: bool = False) -> None:
        self.config = config
        self.validation = validation
        self.records = RecordMixture(config, validation=validation)
        self._active: TextMixtureIterator | None = None
        self._pending_state: dict[str, Any] | None = None

    def __iter__(self) -> Iterator[str]:
        if self._pending_state is not None:
            self.records.load_state_dict(self._pending_state["records"])
        iterator = TextMixtureIterator(self, iter(self.records))
        if self._pending_state is not None:
            iterator.load_state_dict(self._pending_state)
            self._pending_state = None
        self._active = iterator
        return iterator

    def state_dict(self) -> dict[str, Any]:
        if self._active is not None:
            return self._active.state_dict()
        if self._pending_state is not None:
            return deepcopy(self._pending_state)
        rng = random.Random(self.config.seed + (1 if self.validation else 0))
        return {
            "schema_version": 1,
            "rng": rng.getstate(),
            "records": self.records.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if self._active is not None:
            raise RuntimeError("Load text mixture state before creating its iterator")
        self._pending_state = deepcopy(state)
