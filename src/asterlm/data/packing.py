from __future__ import annotations

from collections.abc import Iterator

import torch
from torch.utils.data import IterableDataset

from asterlm.config import DataConfig
from .mixture import RecordMixture, TextMixture
from .tokenizer import AsterTokenizer, normalize_messages


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

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        buffer: list[int] = []
        for text in TextMixture(self.data_config, validation=self.validation):
            ids = self.tokenizer.encode(text)
            if not ids:
                continue
            buffer.extend(ids)
            if self.data_config.add_eos_between_documents:
                buffer.append(self.eos_id)
            while len(buffer) >= self.sequence_length + 1:
                chunk = buffer[: self.sequence_length + 1]
                del buffer[: self.sequence_length]
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)
                # Do not spend loss on the arbitrary transition from one document's
                # EOS marker to the first token sampled from an unrelated document.
                if self.data_config.mask_cross_document_loss:
                    labels = labels.masked_fill(input_ids.eq(self.eos_id), self.ignore_index)
                yield {"input_ids": input_ids, "labels": labels}


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
        id_buffer: list[int] = []
        mask_buffer: list[bool] = []
        for record, source in self.records:
            item = self._conversation(record, source)
            if item is None:
                continue
            ids, mask = item
            id_buffer.extend(ids)
            mask_buffer.extend(mask)
            while len(id_buffer) >= self.sequence_length + 1:
                chunk_ids = id_buffer[: self.sequence_length + 1]
                chunk_mask = mask_buffer[: self.sequence_length + 1]
                del id_buffer[: self.sequence_length]
                del mask_buffer[: self.sequence_length]
                labels = [token if allowed else self.ignore_index for token, allowed in zip(chunk_ids[1:], chunk_mask[1:])]
                if all(label == self.ignore_index for label in labels):
                    continue
                yield {
                    "input_ids": torch.tensor(chunk_ids[:-1], dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
