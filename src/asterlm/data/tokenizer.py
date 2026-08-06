from __future__ import annotations

from pathlib import Path
from typing import Any

CORE_SPECIAL_TOKENS = [
    "<|pad|>",
    "<|endoftext|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|tool|>",
    "<|end|>",
    "<|fim_prefix|>",
    "<|fim_suffix|>",
    "<|fim_middle|>",
]

REASONING_SPECIAL_TOKENS = [
    "<|thinking|>",
    "<|direct|>",
    "<think>",
    "</think>",
    "<answer>",
    "</answer>",
]

SPECIAL_TOKENS = CORE_SPECIAL_TOKENS + REASONING_SPECIAL_TOKENS


class AsterTokenizer:
    def __init__(self, path: str | Path) -> None:
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise ImportError("Install `tokenizers` to load AsterLM tokenizers") from exc
        self.path = str(path)
        self.backend = Tokenizer.from_file(self.path)
        self._special_ids = {token: self.backend.token_to_id(token) for token in SPECIAL_TOKENS}
        missing = [token for token in CORE_SPECIAL_TOKENS if self._special_ids[token] is None]
        if missing:
            raise ValueError(f"Tokenizer is missing required special tokens: {missing}")

    @property
    def vocab_size(self) -> int:
        return self.backend.get_vocab_size(with_added_tokens=True)

    def token_to_id(self, token: str) -> int:
        value = self.backend.token_to_id(token)
        if value is None:
            raise KeyError(f"Unknown token: {token}")
        return int(value)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self.backend.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return self.backend.decode(ids, skip_special_tokens=skip_special_tokens)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.backend.save(str(path))


def normalize_messages(messages: Any) -> list[dict[str, str]]:
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    normalized: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", message.get("from", "user"))).strip().lower()
        role = {"human": "user", "gpt": "assistant", "bot": "assistant"}.get(role, role)
        content = message.get("content", message.get("value", ""))
        if isinstance(content, list):
            pieces = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    pieces.append(str(part.get("text", "")))
                elif isinstance(part, str):
                    pieces.append(part)
            content = "\n".join(pieces)
        normalized.append({"role": role, "content": str(content)})
    return normalized


def format_chat(messages: list[dict[str, str]], add_generation_prompt: bool = False) -> str:
    chunks: list[str] = []
    for message in normalize_messages(messages):
        role = message["role"] if message["role"] in {"system", "user", "assistant", "tool"} else "user"
        chunks.append(f"<|{role}|>\n{message['content']}<|end|>\n")
    if add_generation_prompt:
        chunks.append("<|assistant|>\n")
    return "".join(chunks)
