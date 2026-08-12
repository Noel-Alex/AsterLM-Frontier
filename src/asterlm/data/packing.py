from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from typing import Any

import torch
from torch.utils.data import IterableDataset

from asterlm.config import DataConfig

from .mixture import RecordMixture, TextMixture
from .tokenizer import AsterTokenizer, normalize_messages


class PackedTokenIterator:
    def __init__(self, parent: PackedTokenDataset, texts: Iterator[str]) -> None:
        self.parent = parent
        self.texts = texts
        self.buffer: list[int] = []

    def __iter__(self) -> PackedTokenIterator:
        return self

    def __next__(self) -> dict[str, torch.Tensor]:
        while len(self.buffer) < self.parent.sequence_length + 1:
            text = next(self.texts)
            ids = self.parent.tokenizer.encode(text)
            if not ids:
                continue
            self.buffer.extend(ids)
            if self.parent.data_config.add_eos_between_documents:
                self.buffer.append(self.parent.eos_id)
        chunk = self.buffer[: self.parent.sequence_length + 1]
        del self.buffer[: self.parent.sequence_length]
        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        labels = torch.tensor(chunk[1:], dtype=torch.long)
        if self.parent.data_config.mask_cross_document_loss:
            labels = labels.masked_fill(input_ids.eq(self.parent.eos_id), self.parent.ignore_index)
        return {"input_ids": input_ids, "labels": labels}

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "pretrain",
            "sequence_length": self.parent.sequence_length,
            "buffer": list(self.buffer),
            "text_mixture": self.parent.texts.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if (
            state.get("schema_version") != 1
            or state.get("kind") != "pretrain"
            or int(state.get("sequence_length")) != self.parent.sequence_length
        ):
            raise ValueError("Packed-token cursor does not match this dataset")
        self.buffer = [int(value) for value in state["buffer"]]


class SFTPackedIterator:
    def __init__(
        self,
        parent: SFTPackedDataset,
        records: Iterator[tuple[dict[str, Any], Any]],
    ) -> None:
        self.parent = parent
        self.records = records
        self.id_buffer: list[int] = []
        self.mask_buffer: list[bool] = []

    def __iter__(self) -> SFTPackedIterator:
        return self

    def __next__(self) -> dict[str, torch.Tensor]:
        while True:
            while len(self.id_buffer) < self.parent.sequence_length + 1:
                record, source = next(self.records)
                item = self.parent._conversation(record, source)
                if item is None:
                    continue
                ids, mask = item
                self.id_buffer.extend(ids)
                self.mask_buffer.extend(mask)
            chunk_ids = self.id_buffer[: self.parent.sequence_length + 1]
            chunk_mask = self.mask_buffer[: self.parent.sequence_length + 1]
            del self.id_buffer[: self.parent.sequence_length]
            del self.mask_buffer[: self.parent.sequence_length]
            labels = [
                token if allowed else self.parent.ignore_index
                for token, allowed in zip(chunk_ids[1:], chunk_mask[1:], strict=True)
            ]
            if all(label == self.parent.ignore_index for label in labels):
                continue
            return {
                "input_ids": torch.tensor(chunk_ids[:-1], dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "sft",
            "sequence_length": self.parent.sequence_length,
            "id_buffer": list(self.id_buffer),
            "mask_buffer": list(self.mask_buffer),
            "record_mixture": self.parent.records.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if (
            state.get("schema_version") != 1
            or state.get("kind") != "sft"
            or int(state.get("sequence_length")) != self.parent.sequence_length
        ):
            raise ValueError("SFT packer cursor does not match this dataset")
        self.id_buffer = [int(value) for value in state["id_buffer"]]
        self.mask_buffer = [bool(value) for value in state["mask_buffer"]]
        if len(self.id_buffer) != len(self.mask_buffer):
            raise ValueError("SFT packer token/mask buffer lengths differ")


class PackedTokenDataset(IterableDataset):
    """Greedily packs streamed documents into fixed next-token training examples."""

    def __init__(
        self,
        tokenizer: AsterTokenizer,
        data_config: DataConfig,
        sequence_length: int,
        validation: bool = False,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.data_config = data_config
        self.sequence_length = sequence_length
        self.validation = validation
        self.ignore_index = ignore_index
        self.eos_id = tokenizer.token_to_id("<|endoftext|>")
        self.texts = TextMixture(self.data_config, validation=self.validation)
        self._active: PackedTokenIterator | None = None
        self._pending_state: dict[str, Any] | None = None

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        if self._pending_state is not None:
            self.texts.load_state_dict(self._pending_state["text_mixture"])
        iterator = PackedTokenIterator(self, iter(self.texts))
        if self._pending_state is not None:
            iterator.load_state_dict(self._pending_state)
            self._pending_state = None
        self._active = iterator
        return iterator

    def state_dict(self) -> dict[str, Any]:
        if self._active is None:
            if self._pending_state is not None:
                return deepcopy(self._pending_state)
            raise RuntimeError("Packed dataset iterator has not been created")
        return self._active.state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if self._active is not None:
            raise RuntimeError("Load packed dataset state before creating its iterator")
        self._pending_state = deepcopy(state)


class SFTPackedDataset(IterableDataset):
    """Packs conversations while masking non-assistant target tokens."""

    def __init__(
        self,
        tokenizer: AsterTokenizer,
        data_config: DataConfig,
        sequence_length: int,
        validation: bool = False,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.records = RecordMixture(data_config, validation=validation)
        self.sequence_length = sequence_length
        self.ignore_index = ignore_index
        self.eos_id = tokenizer.token_to_id("<|endoftext|>")
        self.end_id = tokenizer.token_to_id("<|end|>")
        self._active: SFTPackedIterator | None = None
        self._pending_state: dict[str, Any] | None = None

    def _conversation(self, record: dict, source) -> tuple[list[int], list[bool]] | None:
        messages = record.get(source.messages_field)
        if not isinstance(messages, list):
            return None
        normalized = normalize_messages(messages)
        system = record.get("system")
        chat_kwargs = record.get("chat_template_kwargs")
        if system is None and isinstance(chat_kwargs, dict):
            system = chat_kwargs.get("system") or chat_kwargs.get("system_prompt")
        if system and (not normalized or normalized[0]["role"] != "system"):
            normalized.insert(0, {"role": "system", "content": str(system)})
        ids: list[int] = []
        trainable: list[bool] = []
        for message in normalized:
            role = message["role"] if message["role"] in {"system", "user", "assistant", "tool"} else "user"
            prefix = self.tokenizer.encode(f"<|{role}|>\n")
            content = self.tokenizer.encode(message["content"])
            suffix = [self.end_id] + self.tokenizer.encode("\n")
            ids.extend(prefix)
            trainable.extend([False] * len(prefix))
            ids.extend(content)
            trainable.extend([role == "assistant"] * len(content))
            ids.extend(suffix)
            trainable.extend([role == "assistant"] * len(suffix))
        ids.append(self.eos_id)
        trainable.append(True)
        return ids, trainable

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        if self._pending_state is not None:
            self.records.load_state_dict(self._pending_state["record_mixture"])
        iterator = SFTPackedIterator(self, iter(self.records))
        if self._pending_state is not None:
            iterator.load_state_dict(self._pending_state)
            self._pending_state = None
        self._active = iterator
        return iterator

    def state_dict(self) -> dict[str, Any]:
        if self._active is None:
            if self._pending_state is not None:
                return deepcopy(self._pending_state)
            raise RuntimeError("SFT dataset iterator has not been created")
        return self._active.state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if self._active is not None:
            raise RuntimeError("Load SFT dataset state before creating its iterator")
        self._pending_state = deepcopy(state)
