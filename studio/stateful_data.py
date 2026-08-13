from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.data import IterableDataset

from asterlm.config import DataConfig
from asterlm.data.mixture import (
    _fim_transform,
    _is_local_data_file,
    _iter_local_file,
    _quality_ok,
    record_to_text,
)
from asterlm.data.tokenizer import AsterTokenizer


class StatefulLocalPackedDataset(IterableDataset):
    """Checkpointable single-process packed dataset for local cleaned corpora.

    It preserves the same source-weight sampling and local-file traversal ideas
    as Aster's ordinary TextMixture/PackedTokenDataset, but makes the state
    explicit so a long run does not need to replay every prior microbatch.

    Resume work is bounded by reopening the *current* file for each source and
    skipping the already-consumed records in that file. The residual token
    packing buffer and both RNG states are serialized exactly.
    """

    VERSION = 1

    def __init__(
        self,
        tokenizer: AsterTokenizer,
        data_config: DataConfig,
        sequence_length: int,
        *,
        validation: bool = False,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.data_config = data_config
        self.sequence_length = int(sequence_length)
        self.validation = bool(validation)
        self.ignore_index = int(ignore_index)
        self.eos_id = tokenizer.token_to_id("<|endoftext|>")

        self.sources = (
            list(data_config.validation_sources)
            if validation
            else list(data_config.sources)
        )
        if not self.sources:
            raise ValueError("No local sources configured")
        if any(float(source.weight) <= 0 for source in self.sources):
            raise ValueError("All local source weights must be positive")

        self._files: list[list[Path]] = []
        for source in self.sources:
            root = Path(source.path)
            if not root.exists():
                raise FileNotFoundError(
                    f"Studio stateful pretraining requires local cleaned data; missing {root}"
                )
            paths = sorted(root.rglob("*")) if root.is_dir() else [root]
            files = [
                item
                for item in paths
                if item.is_file() and _is_local_data_file(item)
            ]
            if not files:
                raise RuntimeError(f"No supported local data files under {root}")
            self._files.append(files)

        seed = int(data_config.seed)
        # Match the stock single-worker mixture/FIM seeds.
        self._mixture_rng = random.Random(seed)
        self._fim_rng = random.Random(seed + (1 if validation else 0))
        self._weights = [float(source.weight) for source in self.sources]
        self._cursors = [
            {"file_index": 0, "record_index": 0, "cycles": 0}
            for _ in self.sources
        ]
        self._source_iterators: list[Iterator[dict[str, Any]] | None] = [
            None for _ in self.sources
        ]
        self._buffer: list[int] = []

    def _signature(self) -> dict[str, Any]:
        return {
            "sequence_length": self.sequence_length,
            "validation": self.validation,
            "sources": [
                {
                    # Use config spelling + relative file names/sizes rather than
                    # absolute checkout paths so a Hub checkpoint can be
                    # restored on another machine after copying the same cleaned
                    # corpus to the same configured relative location.
                    "path": str(source.path),
                    "text_field": source.text_field,
                    "weight": float(source.weight),
                    "fim_rate": float(source.fim_rate),
                    "files": [
                        {
                            "path": str(
                                path.relative_to(Path(source.path))
                                if Path(source.path).is_dir()
                                else Path(path.name)
                            ),
                            "size": path.stat().st_size,
                        }
                        for path in files
                    ],
                }
                for source, files in zip(self.sources, self._files)
            ],
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "signature": self._signature(),
            "mixture_rng": self._mixture_rng.getstate(),
            "fim_rng": self._fim_rng.getstate(),
            "cursors": copy.deepcopy(self._cursors),
            "buffer": list(self._buffer),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state.get("version", -1)) != self.VERSION:
            raise RuntimeError(
                f"Unsupported Studio packed-data state version={state.get('version')}"
            )
        if state.get("signature") != self._signature():
            raise RuntimeError(
                "Studio checkpoint data signature does not match the current local "
                "cleaned files/data config. Refusing an inexact fast resume."
            )
        cursors = state.get("cursors")
        if not isinstance(cursors, list) or len(cursors) != len(self.sources):
            raise RuntimeError("Studio checkpoint has an invalid source cursor list")
        self._cursors = copy.deepcopy(cursors)
        self._buffer = [int(token) for token in state.get("buffer", [])]
        self._mixture_rng.setstate(state["mixture_rng"])
        self._fim_rng.setstate(state["fim_rng"])
        self._source_iterators = [None for _ in self.sources]

    def _make_source_iterator(self, source_index: int) -> Iterator[dict[str, Any]]:
        source = self.sources[source_index]
        files = self._files[source_index]
        cursor = self._cursors[source_index]

        # One generator instance represents exactly the remainder of one local
        # source pass. RecordMixture normally restarts an exhausted source and
        # consumes one rng.randrange() while deriving the unused local-source
        # seed; _next_record mirrors that RNG advance.
        while int(cursor["file_index"]) < len(files):
            file_index = int(cursor["file_index"])
            path = files[file_index]
            skip = int(cursor.get("record_index", 0))

            for record_index, record in enumerate(_iter_local_file(path, source)):
                if record_index < skip:
                    continue
                # Cursor always names the *next* record before the record is
                # exposed to the mixture.
                cursor["record_index"] = record_index + 1
                if isinstance(record, dict):
                    yield record

            # A file may legitimately have zero records. Either way the next
            # attempt advances to the next finalized local file.
            cursor["file_index"] = file_index + 1
            cursor["record_index"] = 0

    def _next_record(self, source_index: int) -> dict[str, Any]:
        while True:
            iterator = self._source_iterators[source_index]
            if iterator is None:
                iterator = self._make_source_iterator(source_index)
                self._source_iterators[source_index] = iterator
            try:
                return next(iterator)
            except StopIteration:
                cursor = self._cursors[source_index]
                cursor["file_index"] = 0
                cursor["record_index"] = 0
                cursor["cycles"] = int(cursor.get("cycles", 0)) + 1
                # Match RecordMixture's RNG evolution on local-source restart.
                self._mixture_rng.randrange(1_000_000)
                self._source_iterators[source_index] = None

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        while True:
            # Drain any already-packed samples *before* consuming another source
            # record. This makes a freshly reconstructed iterator equivalent to
            # the suspended generator at a checkpoint, including when one very
            # long document produced several consecutive training samples.
            if len(self._buffer) >= self.sequence_length + 1:
                chunk = self._buffer[: self.sequence_length + 1]
                del self._buffer[: self.sequence_length]
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)
                if self.data_config.mask_cross_document_loss:
                    labels = labels.masked_fill(
                        input_ids.eq(self.eos_id),
                        self.ignore_index,
                    )
                yield {"input_ids": input_ids, "labels": labels}
                continue

            index = self._mixture_rng.choices(
                range(len(self.sources)),
                weights=self._weights,
                k=1,
            )[0]
            source = self.sources[index]
            record = self._next_record(index)
            text = record_to_text(record, source)
            if text is None:
                continue
            text = text.strip()
            if not _quality_ok(text, self.data_config):
                continue
            if (
                not self.validation
                and float(source.fim_rate) > 0
                and self._fim_rng.random() < float(source.fim_rate)
            ):
                text = _fim_transform(text, self._fim_rng)

            ids = self.tokenizer.encode(text)
            if not ids:
                continue
            self._buffer.extend(ids)
            if self.data_config.add_eos_between_documents:
                self._buffer.append(self.eos_id)
