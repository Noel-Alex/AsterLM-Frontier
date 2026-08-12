from __future__ import annotations

import hashlib
import math
import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import torch
import torch.nn.functional as F

LONG_CONTEXT_CASE_SCHEMA_VERSION = 1


class TokenizerLike(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...


@dataclass(frozen=True, slots=True)
class RetrievalCase:
    """One token-exact long-context memory probe.

    ``target_sequence_tokens`` includes the prompt and teacher-forced answer. This
    keeps comparisons exact even when tokenizers split synthetic identifiers in
    different ways. Prompt tokens are deliberately not serialized in result files;
    their digest makes each generated case auditable without duplicating many GiB.
    """

    case_id: str
    task: str
    seed: int
    target_sequence_tokens: int
    requested_depth: float
    actual_depth: float
    prompt_ids: tuple[int, ...]
    answer_ids: tuple[int, ...]
    expected_answer: str
    prompt_sha256: str

    @property
    def prompt_tokens(self) -> int:
        return len(self.prompt_ids)

    @property
    def answer_tokens(self) -> int:
        return len(self.answer_ids)

    def manifest(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("prompt_ids")
        payload.pop("answer_ids")
        payload["schema_version"] = LONG_CONTEXT_CASE_SCHEMA_VERSION
        payload["prompt_tokens"] = self.prompt_tokens
        payload["answer_tokens"] = self.answer_tokens
        return payload


def _token_digest(token_ids: tuple[int, ...] | list[int]) -> str:
    digest = hashlib.sha256()
    for token_id in token_ids:
        if token_id < 0:
            raise ValueError("token ids must be non-negative")
        digest.update(int(token_id).to_bytes(8, byteorder="little", signed=False))
    return digest.hexdigest()


def _encoded(tokenizer: TokenizerLike, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def _filler_tokens(
    tokenizer: TokenizerLike,
    *,
    count: int,
    seed: int,
    forbidden: tuple[str, ...],
) -> list[int]:
    rng = random.Random(seed)
    tokens: list[int] = []
    while len(tokens) < count:
        key = f"D-{rng.randrange(10_000_000, 99_999_999)}"
        value = f"V-{rng.randrange(10_000_000, 99_999_999)}"
        line = f"Archive record: key {key} has value {value}.\n"
        if any(item in line for item in forbidden):
            continue
        encoded = _encoded(tokenizer, line)
        if not encoded:
            raise ValueError("tokenizer produced no tokens for long-context filler")
        tokens.extend(encoded)
    return tokens[:count]


def _assemble_case_tokens(
    tokenizer: TokenizerLike,
    *,
    target_sequence_tokens: int,
    seed: int,
    payloads: list[tuple[str, float, list[int]]],
    query_ids: list[int],
    answer_ids: list[int],
    forbidden: tuple[str, ...],
) -> tuple[tuple[int, ...], float]:
    if not payloads or sum(label == "target" for label, _, _ in payloads) != 1:
        raise ValueError("payloads must contain exactly one target fragment")
    for _, depth, tokens in payloads:
        if not 0.0 <= depth <= 1.0:
            raise ValueError("payload depth must be in [0, 1]")
        if not tokens:
            raise ValueError("payload fragments must not be empty")

    fixed_tokens = sum(len(tokens) for _, _, tokens in payloads) + len(query_ids) + len(answer_ids)
    filler_count = target_sequence_tokens - fixed_tokens
    if filler_count < 0:
        raise ValueError(
            f"target_sequence_tokens={target_sequence_tokens} is too small; "
            f"this case needs at least {fixed_tokens} tokens"
        )
    filler = _filler_tokens(
        tokenizer,
        count=filler_count,
        seed=seed ^ 0xA57E_1A5E,
        forbidden=forbidden,
    )

    positioned = sorted(
        (
            min(filler_count, max(0, round(filler_count * depth))),
            index,
            label,
            tokens,
        )
        for index, (label, depth, tokens) in enumerate(payloads)
    )
    context: list[int] = []
    filler_cursor = 0
    target_start = -1
    target_length = -1
    for filler_position, _, label, tokens in positioned:
        context.extend(filler[filler_cursor:filler_position])
        filler_cursor = filler_position
        if label == "target":
            target_start = len(context)
            target_length = len(tokens)
        context.extend(tokens)
    context.extend(filler[filler_cursor:])
    if target_start < 0:
        raise AssertionError("target fragment position was not recorded")

    prompt_ids = (*context, *query_ids)
    if len(prompt_ids) + len(answer_ids) != target_sequence_tokens:
        raise AssertionError("long-context case construction lost exact token accounting")
    actual_depth = target_start / max(len(context) - target_length, 1)
    return prompt_ids, actual_depth


def build_exact_key_case(
    tokenizer: TokenizerLike,
    *,
    target_sequence_tokens: int,
    depth: float,
    seed: int,
) -> RetrievalCase:
    """Build an exact-length associative-retrieval case.

    The answer is scored with teacher forcing, not free-form generation. This makes
    the probe meaningful for base/pretraining checkpoints and avoids conflating
    long-context memory with chat formatting or instruction-following ability.
    """

    if target_sequence_tokens <= 0:
        raise ValueError("target_sequence_tokens must be positive")
    if not 0.0 <= depth <= 1.0:
        raise ValueError("depth must be in [0, 1]")

    rng = random.Random(seed)
    key = f"K-{rng.randrange(10_000_000, 99_999_999)}"
    value = f"V-{rng.randrange(10_000_000, 99_999_999)}"
    payload_ids = _encoded(
        tokenizer,
        f"Authoritative record: key {key} has value {value}.\n",
    )
    query_ids = _encoded(tokenizer, f"Lookup request: key {key} has value")
    answer_text = f" {value}"
    answer_ids = _encoded(tokenizer, answer_text)
    if not payload_ids or not query_ids or not answer_ids:
        raise ValueError("tokenizer produced an empty payload, query, or answer")

    prompt_ids, actual_depth = _assemble_case_tokens(
        tokenizer,
        target_sequence_tokens=target_sequence_tokens,
        seed=seed,
        payloads=[("target", depth, payload_ids)],
        query_ids=query_ids,
        answer_ids=answer_ids,
        forbidden=(key, value),
    )
    answer_tuple = tuple(answer_ids)
    stable = (
        f"exact_key:{seed}:{target_sequence_tokens}:{depth:.8f}:"
        f"{_token_digest(prompt_ids)}:{_token_digest(answer_tuple)}"
    )
    case_id = hashlib.sha256(stable.encode("utf-8")).hexdigest()[:24]
    return RetrievalCase(
        case_id=case_id,
        task="exact_key",
        seed=seed,
        target_sequence_tokens=target_sequence_tokens,
        requested_depth=depth,
        actual_depth=actual_depth,
        prompt_ids=prompt_ids,
        answer_ids=answer_tuple,
        expected_answer=value,
        prompt_sha256=_token_digest(prompt_ids),
    )


def build_repeated_key_case(
    tokenizer: TokenizerLike,
    *,
    target_sequence_tokens: int,
    depth: float,
    seed: int,
) -> RetrievalCase:
    """Build an interference probe with stale and authoritative values for one key."""

    if not 0.0 <= depth <= 1.0:
        raise ValueError("depth must be in [0, 1]")
    rng = random.Random(seed)
    key = f"K-{rng.randrange(10_000_000, 99_999_999)}"
    stale_values = [f"V-{rng.randrange(10_000_000, 99_999_999)}" for _ in range(2)]
    value = f"V-{rng.randrange(10_000_000, 99_999_999)}"
    stale_depths = ((depth + 0.31) % 1.0, (depth + 0.67) % 1.0)
    payloads = [
        (
            f"stale-{index}",
            stale_depth,
            _encoded(
                tokenizer,
                f"Superseded revision {index}: key {key} had value {stale_value}.\n",
            ),
        )
        for index, (stale_depth, stale_value) in enumerate(
            zip(stale_depths, stale_values, strict=True), start=1
        )
    ]
    payloads.append(
        (
            "target",
            depth,
            _encoded(tokenizer, f"Authoritative revision 3: key {key} has value {value}.\n"),
        )
    )
    query_ids = _encoded(tokenizer, f"Lookup authoritative revision: key {key} has value")
    answer_text = f" {value}"
    answer_ids = _encoded(tokenizer, answer_text)
    prompt_ids, actual_depth = _assemble_case_tokens(
        tokenizer,
        target_sequence_tokens=target_sequence_tokens,
        seed=seed,
        payloads=payloads,
        query_ids=query_ids,
        answer_ids=answer_ids,
        forbidden=(key, value, *stale_values),
    )
    answer_tuple = tuple(answer_ids)
    stable = (
        f"repeated_key:{seed}:{target_sequence_tokens}:{depth:.8f}:"
        f"{_token_digest(prompt_ids)}:{_token_digest(answer_tuple)}"
    )
    return RetrievalCase(
        case_id=hashlib.sha256(stable.encode("utf-8")).hexdigest()[:24],
        task="repeated_key",
        seed=seed,
        target_sequence_tokens=target_sequence_tokens,
        requested_depth=depth,
        actual_depth=actual_depth,
        prompt_ids=prompt_ids,
        answer_ids=answer_tuple,
        expected_answer=value,
        prompt_sha256=_token_digest(prompt_ids),
    )


def build_two_hop_case(
    tokenizer: TokenizerLike,
    *,
    target_sequence_tokens: int,
    depth: float,
    seed: int,
) -> RetrievalCase:
    """Build a two-record associative chain split across distant context positions."""

    if not 0.0 <= depth <= 1.0:
        raise ValueError("depth must be in [0, 1]")
    rng = random.Random(seed)
    key = f"K-{rng.randrange(10_000_000, 99_999_999)}"
    alias = f"A-{rng.randrange(10_000_000, 99_999_999)}"
    value = f"V-{rng.randrange(10_000_000, 99_999_999)}"
    second_depth = 1.0 - depth
    if abs(second_depth - depth) < 0.1:
        second_depth = 0.9 if depth < 0.9 else 0.1
    payloads = [
        ("link", depth, _encoded(tokenizer, f"Alias record: key {key} points to alias {alias}.\n")),
        (
            "target",
            second_depth,
            _encoded(tokenizer, f"Value record: alias {alias} has final value {value}.\n"),
        ),
    ]
    query_ids = _encoded(tokenizer, f"Resolve two-hop lookup: key {key} has final value")
    answer_text = f" {value}"
    answer_ids = _encoded(tokenizer, answer_text)
    prompt_ids, actual_depth = _assemble_case_tokens(
        tokenizer,
        target_sequence_tokens=target_sequence_tokens,
        seed=seed,
        payloads=payloads,
        query_ids=query_ids,
        answer_ids=answer_ids,
        forbidden=(key, alias, value),
    )
    answer_tuple = tuple(answer_ids)
    stable = (
        f"two_hop:{seed}:{target_sequence_tokens}:{depth:.8f}:"
        f"{_token_digest(prompt_ids)}:{_token_digest(answer_tuple)}"
    )
    return RetrievalCase(
        case_id=hashlib.sha256(stable.encode("utf-8")).hexdigest()[:24],
        task="two_hop",
        seed=seed,
        target_sequence_tokens=target_sequence_tokens,
        requested_depth=depth,
        actual_depth=actual_depth,
        prompt_ids=prompt_ids,
        answer_ids=answer_tuple,
        expected_answer=value,
        prompt_sha256=_token_digest(prompt_ids),
    )


def build_retrieval_case(
    tokenizer: TokenizerLike,
    *,
    task: str,
    target_sequence_tokens: int,
    depth: float,
    seed: int,
) -> RetrievalCase:
    builders = {
        "exact_key": build_exact_key_case,
        "repeated_key": build_repeated_key_case,
        "two_hop": build_two_hop_case,
    }
    try:
        builder = builders[task]
    except KeyError as exc:
        raise ValueError(f"unknown retrieval task {task!r}; choose from {sorted(builders)}") from exc
    return builder(
        tokenizer,
        target_sequence_tokens=target_sequence_tokens,
        depth=depth,
        seed=seed,
    )


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def score_retrieval_case(
    model: torch.nn.Module,
    case: RetrievalCase,
    *,
    device: str | torch.device,
    prefill_chunk_size: int = 2048,
) -> dict[str, Any]:
    """Score answer-token recall using the model's real incremental cache path."""

    target_device = torch.device(device)
    if prefill_chunk_size <= 0:
        raise ValueError("prefill_chunk_size must be positive")
    max_seq_len = int(model.config.max_seq_len)
    if case.target_sequence_tokens > max_seq_len:
        raise ValueError(
            f"case length {case.target_sequence_tokens} exceeds model max_seq_len={max_seq_len}"
        )

    model.eval()
    cache = model.make_cache()
    prompt = torch.tensor(case.prompt_ids, dtype=torch.long, device=target_device).unsqueeze(0)
    if target_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target_device)
    _sync(target_device)
    prefill_started = time.perf_counter()
    output = None
    for start in range(0, prompt.shape[1], prefill_chunk_size):
        output = model(prompt[:, start : start + prefill_chunk_size], cache=cache, use_cache=True)
    _sync(target_device)
    prefill_seconds = time.perf_counter() - prefill_started
    if output is None or output.logits is None:
        raise RuntimeError("model did not return logits for retrieval prefill")

    logits = output.logits[:, -1]
    answer_nll = 0.0
    greedy_matches = 0
    decode_started = time.perf_counter()
    for index, token_id in enumerate(case.answer_ids):
        token = torch.tensor([token_id], dtype=torch.long, device=target_device)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        answer_nll -= float(log_probs[0, token_id].item())
        greedy_matches += int(int(logits.argmax(dim=-1).item()) == token_id)
        if index + 1 < len(case.answer_ids):
            output = model(token.view(1, 1), cache=cache, use_cache=True)
            if output.logits is None:
                raise RuntimeError("model did not return logits while scoring retrieval answer")
            logits = output.logits[:, -1]
    _sync(target_device)
    decode_seconds = time.perf_counter() - decode_started

    answer_tokens = len(case.answer_ids)
    mean_nll = answer_nll / answer_tokens
    cache_bytes = int(getattr(cache, "num_bytes", 0))
    peak_vram_bytes = (
        int(torch.cuda.max_memory_allocated(target_device)) if target_device.type == "cuda" else 0
    )
    return {
        **case.manifest(),
        "status": "ok",
        "answer_nll": answer_nll,
        "answer_mean_nll": mean_nll,
        "answer_perplexity": math.exp(min(mean_nll, 20.0)),
        "answer_greedy_token_accuracy": greedy_matches / answer_tokens,
        "answer_exact_greedy": greedy_matches == answer_tokens,
        "prefill_seconds": prefill_seconds,
        "prefill_tokens_per_second": case.prompt_tokens / max(prefill_seconds, 1e-12),
        "answer_decode_seconds": decode_seconds,
        "cache_bytes": cache_bytes,
        "peak_vram_bytes": peak_vram_bytes,
    }


def summarize_retrieval_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in results if row.get("status") == "ok"]
    if not completed:
        return {
            "completed_cases": 0,
            "exact_greedy_accuracy": None,
            "by_task": {},
            "by_length": {},
        }

    def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "cases": len(rows),
            "exact_greedy_accuracy": sum(bool(row["answer_exact_greedy"]) for row in rows)
            / len(rows),
            "mean_answer_nll": sum(float(row["answer_mean_nll"]) for row in rows) / len(rows),
            "mean_prefill_tokens_per_second": sum(
                float(row["prefill_tokens_per_second"]) for row in rows
            )
            / len(rows),
            "max_peak_vram_bytes": max(int(row["peak_vram_bytes"]) for row in rows),
            "max_cache_bytes": max(int(row["cache_bytes"]) for row in rows),
        }

    by_length: dict[str, dict[str, Any]] = {}
    for length in sorted({int(row["target_sequence_tokens"]) for row in completed}):
        rows = [row for row in completed if int(row["target_sequence_tokens"]) == length]
        by_length[str(length)] = aggregate(rows)
    by_task: dict[str, dict[str, Any]] = {}
    for task in sorted({str(row["task"]) for row in completed}):
        task_rows = [row for row in completed if str(row["task"]) == task]
        by_task[task] = {
            **aggregate(task_rows),
            "by_length": {
                str(length): aggregate(
                    [
                        row
                        for row in task_rows
                        if int(row["target_sequence_tokens"]) == length
                    ]
                )
                for length in sorted(
                    {int(row["target_sequence_tokens"]) for row in task_rows}
                )
            },
        }
    return {
        "completed_cases": len(completed),
        "exact_greedy_accuracy": sum(bool(row["answer_exact_greedy"]) for row in completed)
        / len(completed),
        "mean_answer_nll": sum(float(row["answer_mean_nll"]) for row in completed)
        / len(completed),
        "by_task": by_task,
        "by_length": by_length,
    }
