#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create resumable teacher SFT data from any OpenAI-compatible chat endpoint"
    )
    parser.add_argument("--input", required=True, help="JSONL; each row must contain a prompt field")
    parser.add_argument("--output", required=True)
    parser.add_argument("--endpoint", required=True, help="Base URL, e.g. http://localhost:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--prompt-field", default="prompt")
    parser.add_argument("--system", default="Answer accurately, clearly, and with rigorous reasoning.")
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--retries", type=int, default=5)
    args = parser.parse_args()
    try:
        import httpx
    except ImportError as exc:
        raise SystemExit("Install httpx: pip install httpx") from exc

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    if output.exists():
        completed = sum(1 for _ in output.open("r", encoding="utf-8"))
    headers = {"Content-Type": "application/json"}
    key = os.environ.get(args.api_key_env)
    if key:
        headers["Authorization"] = f"Bearer {key}"
    rows = [json.loads(line) for line in Path(args.input).open("r", encoding="utf-8") if line.strip()]
    with httpx.Client(timeout=300) as client, output.open("a", encoding="utf-8") as writer:
        for index, row in enumerate(rows[completed:], start=completed):
            prompt = str(row[args.prompt_field])
            payload = {
                "model": args.model,
                "messages": [
                    {"role": "system", "content": args.system},
                    {"role": "user", "content": prompt},
                ],
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
            }
            error = None
            for attempt in range(args.retries):
                try:
                    response = client.post(args.endpoint.rstrip("/") + "/chat/completions", headers=headers, json=payload)
                    response.raise_for_status()
                    answer = response.json()["choices"][0]["message"]["content"]
                    out = {
                        **row,
                        "messages": [
                            {"role": "system", "content": args.system},
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": answer},
                        ],
                        "teacher_model": args.model,
                    }
                    writer.write(json.dumps(out, ensure_ascii=False) + "\n")
                    writer.flush()
                    error = None
                    break
                except Exception as exc:  # endpoint/network failures are retried and recorded
                    error = exc
                    time.sleep(min(30, 2**attempt))
            if error is not None:
                raise RuntimeError(f"Failed at row {index}") from error
            if (index + 1) % 10 == 0:
                print(f"completed {index + 1}/{len(rows)}")


if __name__ == "__main__":
    main()
