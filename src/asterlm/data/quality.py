from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Iterable

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WS_RE = re.compile(r"[ \t]+")
_EMAIL_RE = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d .()\-]{7,}\d)(?!\d)")
_IPV4_RE = re.compile(r"(?<!\d)(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}(?!\d)")
_PRIVATE_KEY_RE = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----")
_AWS_KEY_RE = re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")
_GITHUB_TOKEN_RE = re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{30,255}\b")
_GENERIC_SECRET_RE = re.compile(
    r"(?i)\b(?:api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token|password)\b"
    r"\s*[:=]\s*['\"]?[A-Za-z0-9/+_.=-]{16,}"
)
_WORD_RE = re.compile(r"[\w']+", re.UNICODE)


@dataclass(slots=True)
class QualityDecision:
    keep: bool
    reason: str
    normalized_text: str
    metrics: dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_RE.sub("", text)
    lines = [_WS_RE.sub(" ", line).rstrip() for line in text.split("\n")]
    # Preserve paragraph structure while preventing pathological blank-line runs.
    out: list[str] = []
    blanks = 0
    for line in lines:
        if line:
            blanks = 0
            out.append(line)
        else:
            blanks += 1
            if blanks <= 2:
                out.append("")
    return "\n".join(out).strip()


def redact_pii(text: str) -> tuple[str, int]:
    count = 0
    for pattern, replacement in (
        (_EMAIL_RE, "<EMAIL>"),
        (_PHONE_RE, "<PHONE>"),
        (_IPV4_RE, "<IPV4>"),
    ):
        text, found = pattern.subn(replacement, text)
        count += found
    return text, count


def contains_secret(text: str) -> bool:
    return any(
        pattern.search(text)
        for pattern in (_PRIVATE_KEY_RE, _AWS_KEY_RE, _GITHUB_TOKEN_RE, _GENERIC_SECRET_RE)
    )


def quality_decision(
    text: str,
    *,
    min_chars: int = 200,
    max_chars: int = 500_000,
    min_alpha_ratio: float = 0.08,
    max_symbol_ratio: float = 0.45,
    min_unique_line_ratio: float = 0.32,
    pii_mode: str = "redact",
    drop_secrets: bool = True,
) -> QualityDecision:
    text = normalize_text(text)
    if not min_chars <= len(text) <= max_chars:
        return QualityDecision(False, "length", text, {"chars": float(len(text))})
    if drop_secrets and contains_secret(text):
        return QualityDecision(False, "secret", text, {"chars": float(len(text))})
    if pii_mode == "redact":
        text, pii_count = redact_pii(text)
    elif pii_mode == "drop":
        _, pii_count = redact_pii(text)
        if pii_count:
            return QualityDecision(False, "pii", text, {"pii_count": float(pii_count)})
    elif pii_mode == "keep":
        pii_count = 0
    else:
        raise ValueError("pii_mode must be redact, drop, or keep")

    visible = [char for char in text if not char.isspace()]
    if not visible:
        return QualityDecision(False, "empty", text, {})
    alpha_ratio = sum(char.isalpha() for char in visible) / len(visible)
    symbol_ratio = sum(not (char.isalnum() or char in ".,;:!?'-_()[]{}<>/\\") for char in visible) / len(visible)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    unique_line_ratio = len(set(lines)) / max(1, len(lines))
    words = _WORD_RE.findall(text.lower())
    word_counts = Counter(words)
    max_word_fraction = (max(word_counts.values()) / len(words)) if words else 1.0
    metrics = {
        "chars": float(len(text)),
        "words": float(len(words)),
        "alpha_ratio": alpha_ratio,
        "symbol_ratio": symbol_ratio,
        "unique_line_ratio": unique_line_ratio,
        "max_word_fraction": max_word_fraction,
        "pii_count": float(pii_count),
    }
    if alpha_ratio < min_alpha_ratio:
        return QualityDecision(False, "low_alpha", text, metrics)
    if symbol_ratio > max_symbol_ratio:
        return QualityDecision(False, "symbol_heavy", text, metrics)
    if len(lines) >= 8 and unique_line_ratio < min_unique_line_ratio:
        return QualityDecision(False, "repeated_lines", text, metrics)
    if len(words) >= 100 and max_word_fraction > 0.25:
        return QualityDecision(False, "repeated_words", text, metrics)
    return QualityDecision(True, "keep", text, metrics)


def exact_digest(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()


def _token_ngrams(text: str, n: int = 3) -> Iterable[str]:
    words = _WORD_RE.findall(text.lower())
    if len(words) < n:
        yield " ".join(words)
        return
    for index in range(len(words) - n + 1):
        yield " ".join(words[index : index + n])


def simhash64(text: str, ngram: int = 3, max_ngrams: int = 20_000) -> int:
    accum = [0] * 64
    for count, gram in enumerate(_token_ngrams(text, ngram)):
        if count >= max_ngrams:
            break
        value = int.from_bytes(hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest(), "little")
        for bit in range(64):
            accum[bit] += 1 if value & (1 << bit) else -1
    out = 0
    for bit, value in enumerate(accum):
        if value >= 0:
            out |= 1 << bit
    return out


def hamming64(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def benchmark_ngrams(texts: Iterable[str], n: int = 13) -> set[bytes]:
    result: set[bytes] = set()
    for text in texts:
        words = _WORD_RE.findall(normalize_text(text).lower())
        for index in range(max(0, len(words) - n + 1)):
            gram = " ".join(words[index : index + n])
            result.add(hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest())
    return result


def contamination_fraction(text: str, hashes: set[bytes], n: int = 13) -> float:
    if not hashes:
        return 0.0
    words = _WORD_RE.findall(text.lower())
    total = max(0, len(words) - n + 1)
    if total == 0:
        return 0.0
    hits = 0
    for index in range(total):
        gram = " ".join(words[index : index + n])
        if hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest() in hashes:
            hits += 1
    return hits / total


def entropy_bits(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())
