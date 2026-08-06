#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from asterlm.config import DataConfig
from asterlm.data.mixture import TextMixture
from asterlm.data.tokenizer import SPECIAL_TOKENS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AsterLM's byte-level BPE tokenizer")
    parser.add_argument("--data", default="configs/data/pretrain_mixture_v1.yaml")
    parser.add_argument("--output", default="artifacts/tokenizer.json")
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--documents", type=int, default=1_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
    except ImportError as exc:
        raise SystemExit("Install dependencies first: pip install -e .") from exc

    config = DataConfig.from_yaml(args.data)
    tokenizer = Tokenizer(models.BPE(unk_token=None, byte_fallback=True))
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC()])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    def limited_iterator():
        for index, text in enumerate(TextMixture(config)):
            if index >= args.documents:
                break
            yield text

    tokenizer.train_from_iterator(limited_iterator(), trainer=trainer, length=args.documents)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output))
    print(f"Saved {tokenizer.get_vocab_size()}-token tokenizer to {output}")


if __name__ == "__main__":
    main()
