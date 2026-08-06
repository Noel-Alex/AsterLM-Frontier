#!/usr/bin/env python
from __future__ import annotations

import argparse

import torch

from asterlm.generation.decode import load_runtime
from asterlm.generation.sampling import apply_repetition_penalty, sample_next
from asterlm.reasoning import ReasoningMode, format_reasoning_prompt


@torch.inference_mode()
def generate_budgeted(model, tokenizer, prompt_ids, args):
    cache = model.make_cache()
    output = None
    for start in range(0, len(prompt_ids), args.prefill_chunk_size):
        chunk = torch.tensor(prompt_ids[start : start + args.prefill_chunk_size], device=args.device).unsqueeze(0)
        output = model(chunk, cache=cache, use_cache=True)
    assert output is not None and output.logits is not None
    logits = output.logits[:, -1]
    history = torch.tensor(prompt_ids, device=args.device)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    end_ids = {tokenizer.token_to_id("<|end|>"), tokenizer.token_to_id("<|endoftext|>")}
    close_think = tokenizer.encode("</think>\n<answer>")
    generated = []
    think_tokens = 0
    closed = args.mode == "direct"

    for _ in range(args.max_new_tokens):
        if not closed and think_tokens >= args.thinking_budget:
            for value in close_think:
                generated.append(value)
                history = torch.cat((history, torch.tensor([value], device=args.device)))
                output = model(torch.tensor([[value]], device=args.device), cache=cache, use_cache=True)
                logits = output.logits[:, -1]
                print(tokenizer.decode([value], skip_special_tokens=False), end="", flush=True)
            closed = True
            continue
        penalized = apply_repetition_penalty(logits[0], history, args.repetition_penalty)
        token = sample_next(
            penalized.unsqueeze(0),
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            min_p=args.min_p,
            generator=generator,
        )
        value = int(token.item())
        generated.append(value)
        text = tokenizer.decode([value], skip_special_tokens=False)
        print(text, end="", flush=True)
        history = torch.cat((history, token.view(1)))
        if not closed:
            think_tokens += 1
            decoded_tail = tokenizer.decode(generated[-16:], skip_special_tokens=False).lower()
            if "</think>" in decoded_tail:
                closed = True
        if value in end_ids:
            break
        output = model(token.view(1, 1), cache=cache, use_cache=True)
        logits = output.logits[:, -1]
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive AsterLM think/direct chat with a hard thinking budget")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", choices=["think", "direct"], default="think")
    parser.add_argument("--thinking-budget", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--min-p", type=float, default=0.02)
    parser.add_argument("--repetition-penalty", type=float, default=1.02)
    parser.add_argument("--prefill-chunk-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--quantization", choices=["none", "int4", "int8"], default="none")
    args = parser.parse_args()
    model, tokenizer = load_runtime(
        args.checkpoint,
        args.tokenizer,
        model_config=args.model,
        device=args.device,
        compile_model=args.compile,
        quantization=args.quantization,
    )
    while True:
        try:
            user = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.lower() in {"/quit", "/exit"}:
            break
        prompt = format_reasoning_prompt(user, ReasoningMode(args.mode), force_open_tag=True)
        prompt_ids = tokenizer.encode(prompt)
        print("Aster> ", end="", flush=True)
        generate_budgeted(model, tokenizer, prompt_ids, args)


if __name__ == "__main__":
    main()
