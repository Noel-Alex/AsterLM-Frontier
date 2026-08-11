from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from asterlm.experiments.long_context import (
    build_exact_key_case,
    score_retrieval_case,
    summarize_retrieval_results,
)


class CharacterTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(character) for character in text]


class FakeCache:
    def __init__(self) -> None:
        self.seen_tokens = 0

    @property
    def num_bytes(self) -> int:
        return self.seen_tokens * 4


class ConstantNextTokenModel(torch.nn.Module):
    def __init__(self, predicted_token: int) -> None:
        super().__init__()
        self.predicted_token = predicted_token
        self.config = SimpleNamespace(max_seq_len=1024)

    def make_cache(self) -> FakeCache:
        return FakeCache()

    def forward(self, input_ids, cache=None, use_cache=False):
        assert cache is not None and use_cache
        cache.seen_tokens += input_ids.shape[1]
        logits = torch.full((1, input_ids.shape[1], 256), -10.0)
        logits[..., self.predicted_token] = 10.0
        return SimpleNamespace(logits=logits)


def test_exact_key_case_is_deterministic_and_token_exact():
    tokenizer = CharacterTokenizer()
    first = build_exact_key_case(
        tokenizer,
        target_sequence_tokens=512,
        depth=0.5,
        seed=7,
    )
    second = build_exact_key_case(
        tokenizer,
        target_sequence_tokens=512,
        depth=0.5,
        seed=7,
    )
    assert first == second
    assert first.case_id == second.case_id
    assert first.prompt_tokens + first.answer_tokens == 512
    assert first.actual_depth == pytest.approx(0.5, abs=0.01)
    assert "prompt_ids" not in first.manifest()
    assert "answer_ids" not in first.manifest()


def test_exact_key_case_rejects_invalid_geometry():
    tokenizer = CharacterTokenizer()
    with pytest.raises(ValueError, match="depth"):
        build_exact_key_case(tokenizer, target_sequence_tokens=512, depth=1.1, seed=7)
    with pytest.raises(ValueError, match="too small"):
        build_exact_key_case(tokenizer, target_sequence_tokens=2, depth=0.5, seed=7)


def test_teacher_forced_score_uses_incremental_cache_and_exact_token_metric():
    case = build_exact_key_case(
        CharacterTokenizer(),
        target_sequence_tokens=512,
        depth=0.5,
        seed=11,
    )
    predicted = case.answer_ids[0]
    model = ConstantNextTokenModel(predicted)
    result = score_retrieval_case(model, case, device="cpu", prefill_chunk_size=37)
    expected_matches = sum(token == predicted for token in case.answer_ids)
    assert result["answer_greedy_token_accuracy"] == expected_matches / len(case.answer_ids)
    assert result["cache_bytes"] == (case.prompt_tokens + case.answer_tokens - 1) * 4
    assert result["prefill_tokens_per_second"] > 0


def test_retrieval_summary_keeps_lengths_separate():
    rows = [
        {
            "status": "ok",
            "target_sequence_tokens": 4096,
            "answer_exact_greedy": True,
            "answer_mean_nll": 1.0,
            "prefill_tokens_per_second": 100.0,
            "peak_vram_bytes": 10,
            "cache_bytes": 5,
        },
        {
            "status": "ok",
            "target_sequence_tokens": 8192,
            "answer_exact_greedy": False,
            "answer_mean_nll": 2.0,
            "prefill_tokens_per_second": 50.0,
            "peak_vram_bytes": 20,
            "cache_bytes": 10,
        },
    ]
    summary = summarize_retrieval_results(rows)
    assert summary["exact_greedy_accuracy"] == 0.5
    assert summary["by_length"]["4096"]["exact_greedy_accuracy"] == 1.0
    assert summary["by_length"]["8192"]["mean_answer_nll"] == 2.0
