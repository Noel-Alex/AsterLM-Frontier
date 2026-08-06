from __future__ import annotations

import gzip
import io
import json
import random
import re
from collections.abc import Iterator
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
        for candidate in ("text", "content", "deepseek_solution", "generated_solution", "solution", "response"):
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
    }:
        return False
    if name.startswith("cursor-") or name.endswith((".partial", ".tmp", ".pkl", ".lock", ".log", ".sqlite", ".db")):
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
    paths = sorted(path.rglob("*")) if path.is_dir() else [path]
    worker = get_worker_info()
    worker_id = 0 if worker is None else worker.id
    num_workers = 1 if worker is None else worker.num_workers
    record_index = 0
    for item in paths:
        if not item.is_file() or not _is_local_data_file(item):
            continue
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


class RecordMixture:
    def __init__(self, config: DataConfig, validation: bool = False) -> None:
        self.config = config
        self.sources = config.validation_sources if validation else config.sources
        if not self.sources:
            raise ValueError("No data sources configured")
        if any(source.weight <= 0 for source in self.sources):
            raise ValueError("All source weights must be positive")

    def __iter__(self) -> Iterator[tuple[dict[str, Any], SourceConfig]]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        rng = random.Random(self.config.seed + 1009 * worker_id)
        weights = [source.weight for source in self.sources]
        iterators = [
            iter_source(source, self.config.seed + i + worker_id, self.config.shuffle_buffer)
            for i, source in enumerate(self.sources)
        ]
        while True:
            index = rng.choices(range(len(self.sources)), weights=weights, k=1)[0]
            try:
                record = next(iterators[index])
            except StopIteration:
                iterators[index] = iter_source(
                    self.sources[index],
                    self.config.seed + index + worker_id + rng.randrange(1_000_000),
                    self.config.shuffle_buffer,
                )
                record = next(iterators[index])
            if isinstance(record, dict):
                yield record, self.sources[index]


class TextMixture:
    def __init__(self, config: DataConfig, validation: bool = False) -> None:
        self.config = config
        self.validation = validation
        self.records = RecordMixture(config, validation=validation)

    def __iter__(self) -> Iterator[str]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        rng = random.Random(self.config.seed + 7919 * worker_id + (1 if self.validation else 0))
        for record, source in self.records:
            text = record_to_text(record, source)
            if text is None:
                continue
            text = text.strip()
            if not _quality_ok(text, self.config):
                continue
            if not self.validation and source.fim_rate > 0 and rng.random() < source.fim_rate:
                text = _fim_transform(text, rng)
            yield text
