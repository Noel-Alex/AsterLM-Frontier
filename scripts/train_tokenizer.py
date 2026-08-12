#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import os
import re
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from asterlm.artifacts import atomic_write_json, sha256_file
from asterlm.config import DataConfig
from asterlm.data.mixture import TextMixture
from asterlm.data.tokenizer import SPECIAL_TOKENS
from asterlm.training.contracts import canonical_data_config_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AsterLM's byte-level BPE tokenizer")
    parser.add_argument("--data", default="configs/data/pretrain_mixture_v1.yaml")
    parser.add_argument("--output", default="artifacts/tokenizer.json")
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--documents", type=int, default=1_000_000)
    parser.add_argument(
        "--manifest",
        default=None,
        help="Seal manifest path (default: tokenizer_manifest.json beside --output)",
    )
    parser.add_argument(
        "--fertility-documents",
        type=int,
        default=1_000,
        help="Documents sampled independently from every source for fertility metrics",
    )
    return parser.parse_args()


def source_fertility(tokenizer, config: DataConfig, documents: int) -> list[dict]:
    rows: list[dict] = []
    for index, source in enumerate(config.sources):
        source_config = replace(source, weight=1.0, fim_rate=0.0)
        isolated = replace(
            config,
            sources=[source_config],
            validation_sources=[],
            seed=config.seed + index * 104_729,
        )
        document_count = 0
        characters = 0
        utf8_bytes = 0
        words = 0
        token_count = 0
        sample_digest = hashlib.sha256()
        error: str | None = None
        try:
            for text in TextMixture(isolated):
                encoded = tokenizer.encode(text)
                encoded_tokens = len(encoded.ids)
                document_count += 1
                characters += len(text)
                utf8 = text.encode("utf-8")
                utf8_bytes += len(utf8)
                words += len(re.findall(r"\S+", text))
                token_count += encoded_tokens
                sample_digest.update(len(utf8).to_bytes(8, "little"))
                sample_digest.update(utf8)
                if document_count >= documents:
                    break
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "source_index": index,
                "path": source.path,
                "name": source.name,
                "revision": source.revision,
                "documents": document_count,
                "characters": characters,
                "utf8_bytes": utf8_bytes,
                "words_whitespace_proxy": words,
                "tokens": token_count,
                "tokens_per_word": token_count / max(1, words),
                "bytes_per_token": utf8_bytes / max(1, token_count),
                "tokens_per_kib": token_count * 1024 / max(1, utf8_bytes),
                "sample_sha256": sample_digest.hexdigest(),
                "error": error,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    if args.documents <= 0 or args.fertility_documents <= 0:
        raise SystemExit("--documents and --fertility-documents must be positive")
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

    training_stats = {"documents": 0, "characters": 0, "utf8_bytes": 0}

    def limited_iterator():
        for text in TextMixture(config):
            if training_stats["documents"] >= args.documents:
                break
            training_stats["documents"] += 1
            training_stats["characters"] += len(text)
            training_stats["utf8_bytes"] += len(text.encode("utf-8"))
            yield text

    tokenizer.train_from_iterator(limited_iterator(), trainer=trainer, length=args.documents)
    if training_stats["documents"] == 0:
        raise RuntimeError("Tokenizer training mixture yielded zero accepted documents")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    tokenizer.save(str(temporary))
    os.replace(temporary, output)

    manifest_path = (
        Path(args.manifest)
        if args.manifest
        else output.with_name(f"{output.stem}_manifest.json")
    )
    fertility = source_fertility(tokenizer, config, args.fertility_documents)
    failed_sources = [row for row in fertility if row["error"] or row["documents"] == 0]
    if failed_sources:
        details = ", ".join(str(row["path"]) for row in failed_sources)
        raise RuntimeError(f"Tokenizer fertility sampling failed for: {details}")
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_config_path": str(Path(args.data).resolve()),
        "data_config_file_sha256": sha256_file(args.data),
        "data_config_sha256": canonical_data_config_sha256(config),
        "tokenizer": {
            "path": str(output.resolve()),
            "sha256": sha256_file(output),
            "vocab_size": tokenizer.get_vocab_size(),
            "special_token_ids": {
                token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS
            },
        },
        "training": {
            **training_stats,
            "requested_documents": args.documents,
            "vocab_size": args.vocab_size,
            "min_frequency": args.min_frequency,
            "normalization": "NFKC",
            "pre_tokenizer": "ByteLevel(add_prefix_space=false,use_regex=true)",
            "byte_fallback": True,
        },
        "fertility_documents_per_source": args.fertility_documents,
        "fertility": fertility,
    }
    atomic_write_json(manifest_path, manifest)
    print(
        f"Saved and sealed {tokenizer.get_vocab_size()}-token tokenizer to {output}; "
        f"manifest={manifest_path}"
    )


if __name__ == "__main__":
    main()
