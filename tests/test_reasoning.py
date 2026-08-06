from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from asterlm.data.tokenizer import REASONING_SPECIAL_TOKENS, SPECIAL_TOKENS
from asterlm.reasoning import (
    RLVRConfig,
    ReasoningMode,
    compute_group_advantages,
    extract_final_answer,
    format_reasoning_prompt,
    rlvr_policy_loss,
    score_completion,
    selected_token_logprobs_from_hidden,
)
from asterlm.reasoning.training import build_rl_sequence, pad_rl_batch
import asterlm.reasoning.verifiers as verifiers


def test_reasoning_special_tokens_are_registered() -> None:
    assert set(REASONING_SPECIAL_TOKENS).issubset(SPECIAL_TOKENS)
    assert {"<|thinking|>", "<|direct|>", "<think>", "</think>", "<answer>", "</answer>"}.issubset(
        REASONING_SPECIAL_TOKENS
    )


def test_prompt_modes_and_final_answer_extraction() -> None:
    thinking = format_reasoning_prompt("What is 2+2?", ReasoningMode.THINK)
    direct = format_reasoning_prompt("What is 2+2?", ReasoningMode.DIRECT)
    assert thinking.endswith("<|thinking|>\n<think>\n")
    assert direct.endswith("<|direct|>\n<answer>\n")
    text = "<think>First I considered 3, then computed carefully.</think>\n<answer>4</answer>"
    assert extract_final_answer(text) == "4"


def test_group_advantages_and_dr_grpo() -> None:
    rewards = torch.tensor([0.0, 1.0, 2.0, 2.0])
    groups = torch.tensor([0, 0, 1, 1])
    standard, stds = compute_group_advantages(rewards, groups, algorithm="gspo")
    dr, _ = compute_group_advantages(rewards, groups, algorithm="dr_grpo")
    assert torch.allclose(standard[:2], torch.tensor([-1.0, 1.0]), atol=1e-5)
    assert torch.allclose(dr[:2], torch.tensor([-0.5, 0.5]))
    assert torch.allclose(dr[2:], torch.zeros(2))
    assert stds.shape == (2,)


def test_all_rlvr_losses_are_finite_and_differentiable() -> None:
    current = torch.tensor([[-1.0, -1.2], [-0.8, -1.4]], requires_grad=True)
    old = torch.tensor([[-1.1, -1.1], [-0.9, -1.3]])
    mask = torch.ones_like(current, dtype=torch.bool)
    advantage = torch.tensor([1.0, -1.0])
    reference = torch.tensor([[-1.15, -1.0], [-1.0, -1.1]])
    for algorithm in ("gspo", "dapo", "grpo", "dr_grpo", "reinforce_baseline"):
        output = rlvr_policy_loss(
            current,
            old,
            mask,
            advantage,
            algorithm=algorithm,
            clip_low=0.2,
            clip_high=0.28,
            reference_token_logps=reference,
            kl_beta=0.001,
        )
        assert torch.isfinite(output.loss)
        output.loss.backward(retain_graph=True)
    assert current.grad is not None and torch.isfinite(current.grad).all()


def test_rl_sequence_alignment_and_padding() -> None:
    record = {
        "prompt_ids": [10, 11, 12],
        "completion_ids": [20, 21],
        "old_token_logps": [-0.2, -0.3],
        "reference_token_logps": [-0.25, -0.35],
        "advantage": 1.5,
        "reward": 1.0,
        "group_id": 7,
    }
    sequence = build_rl_sequence(record)
    assert sequence.input_ids.tolist() == [10, 11, 12, 20]
    assert sequence.targets.tolist() == [11, 12, 20, 21]
    assert sequence.completion_mask.tolist() == [False, False, True, True]
    assert sequence.old_token_logps[sequence.completion_mask].tolist() == pytest_approx([-0.2, -0.3])
    batch = pad_rl_batch([sequence], pad_id=0)
    assert batch["completion_mask"].sum().item() == 2
    assert batch["reference_token_logps"] is not None


def pytest_approx(values: list[float]):
    # Keep this test module independent of pytest's public import in helper annotations.
    import pytest

    return pytest.approx(values)



def test_chunked_selected_logprobs_match_full_projection() -> None:
    class TinyHead(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding_out_proj = torch.nn.Linear(7, 7, bias=False)
            self.lm_head = torch.nn.Linear(7, 13, bias=False)

    torch.manual_seed(7)
    model = TinyHead()
    model.train()
    hidden = torch.randn(2, 9, 7, requires_grad=True)
    targets = torch.randint(0, 13, (2, 9))
    mask = torch.rand(2, 9) > 0.25
    full = model.lm_head(model.embedding_out_proj(hidden))
    expected = full.float().log_softmax(dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1) * mask
    actual = selected_token_logprobs_from_hidden(
        model, hidden, targets, mask, chunk_size=3, checkpoint_chunks=True
    )
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    actual.sum().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()

def test_math_reward_requires_declared_final_answer() -> None:
    cfg = RLVRConfig(iterations=1, prompts_per_iteration=1, group_size=2, max_completion_tokens=128)
    correct = score_completion(
        {"task": "math", "answer": "4", "completion_tokens": 20},
        "<think>Compute two plus two.</think>\n<answer>4</answer>",
        cfg,
    )
    hidden_only = score_completion(
        {"task": "math", "answer": "4", "completion_tokens": 20},
        "<think>The answer is 4 but I never declare it.</think>",
        cfg,
    )
    assert correct.correctness == 1.0 and correct.format == 1.0
    assert hidden_only.correctness == 0.0 and hidden_only.invalid == 1.0


def test_safe_python_stdin_and_function_verifiers(monkeypatch) -> None:
    monkeypatch.setattr(verifiers.shutil, "which", lambda _: None)
    ok, _ = verifiers.verify_python_code(
        "print(int(input()) * 2)",
        [{"input": "3\n", "output": "6\n", "type": "stdin_stdout", "fn_name": None}],
        timeout_seconds=2.0,
        memory_mb=256,
        allow_unsafe=True,
    )
    assert ok
    ok, detail = verifiers.verify_python_code(
        "def add(a, b):\n    return a + b\n",
        [{"input": "[2, 5]", "output": "7", "type": "functional", "fn_name": "add"}],
        timeout_seconds=2.0,
        memory_mb=256,
        allow_unsafe=True,
    )
    assert ok, detail


def test_reasoning_data_converter_extracts_nested_tests(tmp_path: Path) -> None:
    script = Path(__file__).parents[1] / "scripts" / "prepare_reasoning_data.py"
    spec = importlib.util.spec_from_file_location("prepare_reasoning_data", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "code.jsonl"
    source.write_text(
        '{"problem":"Double the input","gold_standard_solution":"```python\\nprint(int(input())*2)\\n```",'
        '"task_type":"verifiable_code","source":"unit",'
        '"verification_info":{"language":"python","test_cases":[{"input":"3\\n","output":"6\\n",'
        '"type":"stdin_stdout","fn_name":null}]}}\n',
        encoding="utf-8",
    )
    sft, rl, stats = module.convert([str(source)])
    assert stats["sft"] == 0 and stats["rl"] == 1
    assert rl[0]["task"] == "code"
    assert rl[0]["tests"][0]["output"] == "6\n"
    assert rl[0]["verification_language"] == "python"


def test_reasoning_trace_gets_mode_token_and_direct_variant(tmp_path: Path) -> None:
    script = Path(__file__).parents[1] / "scripts" / "prepare_reasoning_data.py"
    spec = importlib.util.spec_from_file_location("prepare_reasoning_data_modes", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "trace.jsonl"
    source.write_text(
        '{"messages":[{"role":"user","content":"2+2?"},{"role":"assistant",'
        '"content":"We add the values. Answer: 4"}],"source":"unit"}\n',
        encoding="utf-8",
    )
    sft, rl, stats = module.convert([str(source)], direct_fraction=1.0)
    assert len(sft) == 2 and not rl
    assert sft[0]["messages"][1]["content"].startswith("<|thinking|>\n<think>")
    assert sft[1]["messages"][1]["content"] == "<|direct|>\n<answer>4</answer>"
    assert stats["sft_direct"] == 1


def test_reasoning_reader_ignores_materializer_control_json(tmp_path):
    import json

    from asterlm.reasoning.io import iter_json_records

    (tmp_path / "data.jsonl").write_text(json.dumps({"problem": "2+2"}) + "\n", encoding="utf-8")
    (tmp_path / "state.json").write_text(json.dumps({"problem": "not data"}), encoding="utf-8")
    (tmp_path / "download_manifest.json").write_text(
        json.dumps({"problem": "not data"}), encoding="utf-8"
    )
    assert list(iter_json_records(tmp_path)) == [{"problem": "2+2"}]


def test_preference_response_extracts_last_assistant_message():
    from asterlm.data.preference import _response_text

    response = [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Try again"},
        {"role": "assistant", "content": "Final answer"},
    ]
    assert _response_text(response) == "Final answer"
