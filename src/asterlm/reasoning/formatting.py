from __future__ import annotations

import re
from enum import StrEnum
from typing import Any

from asterlm.data.tokenizer import format_chat, normalize_messages


class ReasoningMode(StrEnum):
    THINK = "think"
    DIRECT = "direct"
    AUTO = "auto"


_BOXED = re.compile(r"\\boxed\s*\{((?:[^{}]|\{[^{}]*\})*)\}", re.DOTALL)
_ANSWER_LINE = re.compile(r"(?:^|\n)\s*(?:final\s+)?answer\s*:\s*(.+?)\s*$", re.IGNORECASE | re.DOTALL)
_ANSWER_INLINE = re.compile(r"(?:final\s+)?answer\s*:\s*([^\n]+?)\s*$", re.IGNORECASE)
_XML_ANSWER = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_THINK = re.compile(r"<think>\s*(.*?)\s*</think>", re.IGNORECASE | re.DOTALL)


def format_reasoning_prompt(
    messages: list[dict[str, str]] | str,
    mode: ReasoningMode | str = ReasoningMode.THINK,
    *,
    force_open_tag: bool = True,
) -> str:
    """Render Aster chat and start either a reasoning or direct response.

    The explicit mode marker lets one checkpoint learn Qwen3-style think/direct
    behavior.  The open XML tag is also emitted during training so inference can
    reliably force reasoning without depending on a natural-language system prompt.
    """

    mode = ReasoningMode(mode)
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    normalized = normalize_messages(messages)
    prefix = format_chat(normalized, add_generation_prompt=True)
    if mode == ReasoningMode.AUTO:
        return prefix
    marker = "<|thinking|>" if mode == ReasoningMode.THINK else "<|direct|>"
    if not force_open_tag:
        return f"{prefix}{marker}\n"
    tag = "<think>\n" if mode == ReasoningMode.THINK else "<answer>\n"
    return f"{prefix}{marker}\n{tag}"


def split_reasoning_answer(text: str) -> tuple[str, str]:
    reasoning_match = _THINK.search(text)
    answer_match = _XML_ANSWER.search(text)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    if answer_match:
        answer = answer_match.group(1).strip()
    else:
        answer = extract_final_answer(text)
    return reasoning, answer


def extract_final_answer(text: str) -> str:
    """Extract only the declared final answer, never an intermediate thought."""

    matches = list(_XML_ANSWER.finditer(text))
    if matches:
        return matches[-1].group(1).strip()
    matches = list(_BOXED.finditer(text))
    if matches:
        return matches[-1].group(1).strip()
    match = _ANSWER_LINE.search(text)
    if match:
        return match.group(1).strip()
    match = _ANSWER_INLINE.search(text)
    if match:
        return match.group(1).strip()
    # If a think section exists, only inspect text after it; this avoids rewarding
    # an intermediate number that happened to appear in the chain of thought.
    close = text.lower().rfind("</think>")
    tail = text[close + len("</think>") :] if close >= 0 else text
    lines = [line.strip() for line in tail.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def ensure_reasoning_markup(content: str, answer: Any | None = None) -> str:
    """Normalize a teacher trace to Aster's think/answer contract."""

    content = str(content).strip()
    if "<think>" in content.lower() and "</think>" in content.lower():
        if "<answer>" in content.lower() or _BOXED.search(content) or _ANSWER_LINE.search(content):
            return content
        final = "" if answer is None else str(answer).strip()
        return f"{content}\n<answer>{final}</answer>" if final else content

    final = str(answer).strip() if answer is not None else extract_final_answer(content)
    reasoning = content
    if final:
        # Remove only a trailing declared final answer.  Mathematical values inside
        # the reasoning are left untouched.
        reasoning = _ANSWER_LINE.sub("", reasoning).strip()
        return f"<think>\n{reasoning}\n</think>\n<answer>{final}</answer>"
    return f"<think>\n{reasoning}\n</think>"
