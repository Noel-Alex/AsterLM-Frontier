#!/usr/bin/env python
from __future__ import annotations

import argparse

import torch

from asterlm.data import format_chat
from asterlm.generation import GenerationConfig, generate, load_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive AsterLM chat")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--system", default="You are a helpful, accurate assistant.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    model, tokenizer = load_runtime(args.checkpoint, args.tokenizer, args.model, args.device)
    eos = tokenizer.token_to_id("<|end|>")
    history = [{"role": "system", "content": args.system}]
    print("AsterLM chat. Commands: /reset, /quit")
    while True:
        try:
            user = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user == "/quit":
            break
        if user == "/reset":
            history = [{"role": "system", "content": args.system}]
            print("history reset")
            continue
        if not user:
            continue
        history.append({"role": "user", "content": user})
        prompt = format_chat(history, add_generation_prompt=True)
        input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=args.device)
        result = generate(
            model,
            input_ids,
            GenerationConfig(max_new_tokens=args.max_new_tokens, eos_token_id=eos),
        )
        answer = tokenizer.decode(result[0, input_ids.shape[1] :].tolist()).replace("<|end|>", "").strip()
        print(f"aster> {answer}")
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
