#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from asterlm.data import format_chat
from asterlm.generation import (
    GenerationConfig,
    download_hub_checkpoint,
    generate,
    list_hub_checkpoints,
    load_runtime,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive AsterLM chat")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint")
    source.add_argument("--hub-repo", help="Private Hugging Face model repository")
    parser.add_argument("--hub-run", default=None)
    parser.add_argument("--hub-selector", default="latest")
    parser.add_argument("--hub-revision", default="main")
    parser.add_argument("--hub-cache-dir", default=None)
    parser.add_argument("--list-hub-checkpoints", action="store_true")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--system", default="You are a helpful, accurate assistant.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    if args.hub_repo and args.list_hub_checkpoints:
        print(
            json.dumps(
                list_hub_checkpoints(args.hub_repo, revision=args.hub_revision),
                indent=2,
            )
        )
        return
    checkpoint = args.checkpoint
    if args.hub_repo:
        checkpoint, identity = download_hub_checkpoint(
            args.hub_repo,
            run=args.hub_run,
            selector=args.hub_selector,
            revision=args.hub_revision,
            cache_dir=args.hub_cache_dir,
        )
        print(f"Loaded Hub checkpoint: {identity}")
    assert checkpoint is not None
    bundled_tokenizer = Path(checkpoint) / "tokenizer.json"
    tokenizer_path = args.tokenizer or str(
        bundled_tokenizer if bundled_tokenizer.is_file() else "artifacts/tokenizer.json"
    )
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    model, tokenizer = load_runtime(checkpoint, tokenizer_path, args.model, device)
    eos = tokenizer.token_to_id("<|end|>")
    history = [{"role": "system", "content": args.system}]
    print(
        "AsterLM chat. Commands: /reset, /quit\n"
        "Note: a pretraining-only checkpoint is a base completion model; reliable "
        "chat behavior requires the later post-training stage."
    )
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
        input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
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
