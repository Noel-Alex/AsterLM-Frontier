#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from asterlm.data.tokenizer import normalize_messages
from asterlm.reasoning.formatting import ensure_reasoning_markup, extract_final_answer
from asterlm.reasoning.io import atomic_write_jsonl, iter_json_records


def stable_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def prompt_from_messages(messages: Any) -> str:
    normalized = normalize_messages(messages)
    users = [message["content"] for message in normalized if message["role"] == "user"]
    return users[-1].strip() if users else ""


def assistant_from_messages(messages: Any) -> str:
    normalized = normalize_messages(messages)
    assistants = [message["content"] for message in normalized if message["role"] == "assistant"]
    return assistants[-1].strip() if assistants else ""


def correct_generation(record: dict[str, Any]) -> str:
    generations = record.get("generations")
    correctness = record.get("correctness_math_verify")
    complete = record.get("is_reasoning_complete")
    if isinstance(generations, list):
        for index, generation in enumerate(generations):
            ok = not isinstance(correctness, list) or index >= len(correctness) or bool(correctness[index])
            done = not isinstance(complete, list) or index >= len(complete) or bool(complete[index])
            if generation and ok and done:
                return str(generation)
        for generation in generations:
            if generation:
                return str(generation)
    return str(record.get("solution", record.get("generation", record.get("gold_standard_solution", ""))))


def ground_truth(record: dict[str, Any]) -> str:
    reward_model = record.get("reward_model")
    if isinstance(reward_model, dict) and reward_model.get("ground_truth") is not None:
        return str(reward_model["ground_truth"])
    for key in ("answer", "solution", "ground_truth", "reference_answer", "gold_standard_solution"):
        value = record.get(key)
        if value is not None and not isinstance(value, (dict, list)):
            return str(value)
    return ""


def task_type(record: dict[str, Any]) -> str:
    value = str(record.get("ability", record.get("task_type", record.get("task", record.get("domain", "math"))))).lower()
    if "code" in value or "python" in value or "program" in value:
        return "code"
    if "choice" in value or "mcq" in value:
        return "multiple_choice"
    return "math" if "math" in value or value in {"stem", "geometry", "algebra"} else value


def convert(inputs: list[str], max_records: int = 0, direct_fraction: float = 0.10):
    seen_sft: set[str] = set()
    seen_rl: set[str] = set()
    sft: list[dict[str, Any]] = []
    rl: list[dict[str, Any]] = []
    stats: dict[str, int] = {"read": 0, "sft": 0, "sft_direct": 0, "rl": 0, "duplicates": 0, "skipped": 0}

    for input_path in inputs:
        for record in iter_json_records(input_path):
            stats["read"] += 1
            messages = record.get("messages", record.get("conversations", record.get("prompt")))
            prompt = ""
            assistant = ""
            teacher_trace = False
            if isinstance(messages, list):
                if messages and isinstance(messages[0], dict) and "role" in messages[0]:
                    prompt = prompt_from_messages(messages)
                    assistant = assistant_from_messages(messages)
                    teacher_trace = bool(assistant)
                elif messages and isinstance(messages[0], dict) and "content" in messages[0]:
                    prompt = prompt_from_messages(messages)
            if not prompt:
                raw_prompt = record.get("problem", record.get("prompt", record.get("question", "")))
                if isinstance(raw_prompt, list):
                    prompt = prompt_from_messages(raw_prompt)
                else:
                    prompt = str(raw_prompt).strip()
            answer = ground_truth(record)
            if not assistant:
                assistant = correct_generation(record)
                teacher_trace = bool(record.get("generations") or record.get("reasoning"))
            source = str(record.get("source", record.get("data_source", record.get("_dataset", "unknown"))))
            task = task_type(record)

            if prompt and assistant and teacher_trace:
                key = stable_id(prompt + "\0" + assistant)
                if key not in seen_sft:
                    seen_sft.add(key)
                    content = ensure_reasoning_markup(assistant, answer or None)
                    content = "<|thinking|>\n" + content
                    sft.append(
                        {
                            "id": key,
                            "messages": [
                                {"role": "user", "content": prompt},
                                {"role": "assistant", "content": content},
                            ],
                            "source": source,
                            "task": task,
                            "mode": "think",
                        }
                    )
                    stats["sft"] += 1
                    final = answer or extract_final_answer(content)
                    selector = int(stable_id(prompt + "\0direct"), 16) / float(16**24 - 1)
                    if final and selector < direct_fraction:
                        direct_key = stable_id(prompt + "\0direct\0" + str(final))
                        sft.append(
                            {
                                "id": direct_key,
                                "messages": [
                                    {"role": "user", "content": prompt},
                                    {
                                        "role": "assistant",
                                        "content": f"<|direct|>\n<answer>{str(final).strip()}</answer>",
                                    },
                                ],
                                "source": source,
                                "task": task,
                                "mode": "direct",
                            }
                        )
                        stats["sft"] += 1
                        stats["sft_direct"] += 1
                else:
                    stats["duplicates"] += 1

            if prompt and answer:
                key = stable_id(prompt)
                if key not in seen_rl:
                    seen_rl.add(key)
                    item: dict[str, Any] = {
                        "id": key,
                        "prompt": prompt,
                        "answer": answer,
                        "task": task,
                        "source": source,
                    }
                    tests = record.get("tests", record.get("official_tests"))
                    verification = record.get("verification_info")
                    if not tests and isinstance(verification, dict):
                        tests = verification.get("test_cases")
                    if isinstance(tests, list) and tests:
                        item["tests"] = tests
                        item["verification_language"] = (
                            verification.get("language") if isinstance(verification, dict) else "python"
                        )
                    rl.append(item)
                    stats["rl"] += 1
                else:
                    stats["duplicates"] += 1
            if not prompt or (not assistant and not answer):
                stats["skipped"] += 1
            if max_records and stats["read"] >= max_records:
                return sft, rl, stats
    return sft, rl, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reasoning SFT and RLVR prompt files")
    parser.add_argument("inputs", nargs="+", help="Materialized JSONL/JSONL.ZST files or directories")
    parser.add_argument("--sft-output", default="data/reasoning/reasoning_sft.jsonl")
    parser.add_argument("--rl-output", default="data/reasoning/rlvr_prompts.jsonl")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--direct-fraction", type=float, default=0.10)
    parser.add_argument("--stats", default="data/reasoning/prepare_stats.json")
    args = parser.parse_args()
    if not 0.0 <= args.direct_fraction <= 1.0:
        raise SystemExit("--direct-fraction must be between 0 and 1")
    sft, rl, stats = convert(args.inputs, args.max_records, args.direct_fraction)
    atomic_write_jsonl(args.sft_output, sft)
    atomic_write_jsonl(args.rl_output, rl)
    Path(args.stats).parent.mkdir(parents=True, exist_ok=True)
    Path(args.stats).write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
