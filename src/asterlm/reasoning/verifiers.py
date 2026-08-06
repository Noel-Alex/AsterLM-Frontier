from __future__ import annotations

import json
import math
import os
import re
import resource
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .config import RLVRConfig
from .formatting import extract_final_answer, split_reasoning_answer


@dataclass(slots=True)
class RewardBreakdown:
    total: float
    correctness: float
    format: float
    reasoning_complete: float
    invalid: float
    repetition: float
    overlong: float
    verifier: str
    extracted_answer: str
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize(value: Any) -> str:
    text = str(value).strip()
    text = text.replace("−", "-").replace("–", "-")
    text = re.sub(r"\s+", " ", text)
    text = text.strip("$ \t\n\r.,;:")
    return text


def _decimal_equal(left: str, right: str) -> bool:
    try:
        a, b = Decimal(left), Decimal(right)
    except InvalidOperation:
        return False
    tolerance = max(Decimal("1e-9"), abs(b) * Decimal("1e-8"))
    return abs(a - b) <= tolerance


def verify_math(prediction: str, reference: Any) -> tuple[bool, str]:
    pred = _normalize(prediction)
    gold = _normalize(reference)
    if not pred or not gold:
        return False, "empty answer"
    if pred.casefold() == gold.casefold() or _decimal_equal(pred, gold):
        return True, "normalized exact match"
    try:
        from math_verify import parse, verify

        parsed_pred = parse(pred)
        parsed_gold = parse(gold)
        if verify(parsed_gold, parsed_pred):
            return True, "math-verify"
    except Exception:
        pass
    try:
        import sympy as sp

        lhs = sp.sympify(pred.replace("^", "**"))
        rhs = sp.sympify(gold.replace("^", "**"))
        if sp.simplify(lhs - rhs) == 0:
            return True, "sympy equivalence"
    except Exception:
        pass
    return False, "no mathematical equivalence"


def verify_multiple_choice(prediction: str, reference: Any) -> tuple[bool, str]:
    pred = _normalize(prediction).upper()
    gold = _normalize(reference).upper()
    pred_match = re.search(r"\b([A-Z])\b", pred)
    gold_match = re.search(r"\b([A-Z])\b", gold)
    pred_value = pred_match.group(1) if pred_match else pred[:1]
    gold_value = gold_match.group(1) if gold_match else gold[:1]
    return pred_value == gold_value, f"choice {pred_value!r} vs {gold_value!r}"


def _limit_child(memory_mb: int, timeout_seconds: float) -> None:
    memory = int(memory_mb * 1024 * 1024)
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    cpu = max(1, int(math.ceil(timeout_seconds)))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 * 1024, 16 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))


def _bwrap_command(workdir: Path, script: str = "main.py") -> list[str]:
    command = [
        "bwrap",
        "--die-with-parent",
        "--new-session",
        "--unshare-all",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
    ]
    for root in ("/usr", "/bin", "/lib", "/lib64", "/etc"):
        if Path(root).exists():
            command.extend(["--ro-bind", root, root])
    command.extend(["--bind", str(workdir), "/work", "--chdir", "/work", "python3", "-I", script])
    return command


def _functional_runner(fn_name: str, raw_input: str) -> str:
    return f"""\
import ast
import importlib.util
import json

spec = importlib.util.spec_from_file_location("aster_candidate", __file__.replace("_aster_test.py", "main.py"))
if spec is None or spec.loader is None:
    raise ImportError("unable to load candidate")
main = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main)

raw = {raw_input!r}
try:
    args = json.loads(raw)
except Exception:
    args = ast.literal_eval(raw)
if not isinstance(args, (list, tuple)):
    args = [args]
fn = getattr(main, {fn_name!r}, None)
if fn is None and hasattr(main, "Solution"):
    fn = getattr(main.Solution(), {fn_name!r})
if fn is None:
    raise AttributeError("missing callable: " + {fn_name!r})
result = fn(*args)
print(json.dumps(result, ensure_ascii=False, sort_keys=True))
"""


def _normalized_program_output(value: str) -> str:
    return "\n".join(line.rstrip() for line in value.strip().splitlines())


def verify_python_code(
    code: str,
    tests: list[dict[str, Any]],
    *,
    timeout_seconds: float,
    memory_mb: int,
    allow_unsafe: bool,
) -> tuple[bool, str]:
    if not tests:
        return False, "no executable tests"
    has_bwrap = shutil.which("bwrap") is not None
    if not has_bwrap and not allow_unsafe:
        return False, "bubblewrap unavailable; unsafe code execution refused"
    with tempfile.TemporaryDirectory(prefix="aster-verifier-") as tmp:
        root = Path(tmp)
        (root / "main.py").write_text(code, encoding="utf-8")
        for index, test in enumerate(tests):
            stdin = str(test.get("input", ""))
            expected = str(test.get("output", test.get("expected", ""))).strip()
            fn_name = test.get("fn_name")
            test_type = str(test.get("type", "stdin_stdout"))
            script = "main.py"
            if fn_name or test_type not in {"stdin_stdout", "standard_input"}:
                if not fn_name:
                    return False, f"test {index} has unsupported type {test_type!r}"
                script = "_aster_test.py"
                (root / script).write_text(_functional_runner(str(fn_name), stdin), encoding="utf-8")
            command = _bwrap_command(root, script) if has_bwrap else ["python3", "-I", str(root / script)]
            try:
                result = subprocess.run(
                    command,
                    input=stdin if script == "main.py" else "",
                    text=True,
                    capture_output=True,
                    timeout=timeout_seconds,
                    cwd=None if has_bwrap else root,
                    env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"},
                    preexec_fn=lambda: _limit_child(memory_mb, timeout_seconds),
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return False, f"test {index} timed out"
            if result.returncode != 0:
                return False, f"test {index} exited {result.returncode}: {result.stderr[-400:]}"
            actual = _normalized_program_output(result.stdout)
            expected_normalized = _normalized_program_output(expected)
            if script != "main.py":
                try:
                    actual = json.dumps(json.loads(actual), ensure_ascii=False, sort_keys=True)
                    expected_normalized = json.dumps(json.loads(expected_normalized), ensure_ascii=False, sort_keys=True)
                except Exception:
                    pass
            if actual != expected_normalized:
                return False, f"test {index} output mismatch"
    return True, "all executable tests passed"


def _extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    return blocks[-1].strip() if blocks else text.strip()


def _repetition_score(text: str, n: int = 8) -> float:
    tokens = text.split()
    if len(tokens) < n * 2:
        return 0.0
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    unique = len(set(grams))
    return max(0.0, 1.0 - unique / max(1, len(grams)))


def score_completion(record: dict[str, Any], completion: str, config: RLVRConfig) -> RewardBreakdown:
    reasoning, answer = split_reasoning_answer(completion)
    task = str(record.get("task", record.get("ability", "math"))).lower()
    reference = record.get("answer", record.get("ground_truth", record.get("solution", "")))
    verifier = "exact"
    detail = ""
    correct = False
    if task in {"math", "mathematics", "stem", "algebra", "geometry"}:
        correct, detail = verify_math(answer, reference)
        verifier = "math"
    elif task in {"multiple_choice", "mcq"}:
        correct, detail = verify_multiple_choice(answer, reference)
        verifier = "multiple_choice"
    elif task in {"python", "code", "coding"} and record.get("tests"):
        correct, detail = verify_python_code(
            _extract_code(answer or completion),
            list(record["tests"]),
            timeout_seconds=config.code_timeout_seconds,
            memory_mb=config.code_memory_mb,
            allow_unsafe=config.allow_unsafe_code_verifier,
        )
        verifier = "python_tests"
    else:
        correct = _normalize(answer).casefold() == _normalize(reference).casefold() and bool(_normalize(answer))
        detail = "normalized exact match" if correct else "exact mismatch"

    has_think = "<think>" in completion.lower() and "</think>" in completion.lower()
    has_answer = bool(answer) and (
        "<answer>" in completion.lower()
        or "\\boxed" in completion
        or re.search(r"(?:^|\n)\s*(?:final\s+)?answer\s*:", completion, re.IGNORECASE)
    )
    format_score = 1.0 if (has_answer and (has_think or not config.require_reasoning_tags)) else 0.0
    reasoning_complete = 1.0 if has_think and len(reasoning.split()) >= 4 else 0.0
    invalid = 1.0 if not answer else 0.0
    repetition = _repetition_score(reasoning)
    token_count = int(record.get("completion_tokens", 0))
    overlong = 0.0
    if token_count > config.max_completion_tokens - config.overlong_buffer_tokens:
        start = max(1, config.max_completion_tokens - config.overlong_buffer_tokens)
        overlong = min(1.0, (token_count - start) / max(1, config.overlong_buffer_tokens))

    total = (
        config.correctness_weight * float(correct)
        + config.format_weight * format_score
        + config.reasoning_complete_weight * reasoning_complete
        - config.invalid_penalty * invalid
        - config.repetition_penalty_reward * repetition
        - config.overlong_penalty * overlong
    )
    total = min(config.reward_clip_max, max(config.reward_clip_min, total))
    return RewardBreakdown(
        total=total,
        correctness=float(correct),
        format=format_score,
        reasoning_complete=reasoning_complete,
        invalid=invalid,
        repetition=repetition,
        overlong=overlong,
        verifier=verifier,
        extracted_answer=answer,
        detail=detail,
    )
