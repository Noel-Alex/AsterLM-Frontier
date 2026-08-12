#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import random
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq
import yaml

from asterlm.data.tokenizer import SPECIAL_TOKENS, AsterTokenizer

ROOT = Path.cwd().resolve()
DEFAULT_RUN = ROOT / "runs/frontier-vnext2/proxy-data"
DEFAULT_TOKENIZER = ROOT / "artifacts/tokenizer_proxy.json"

SOURCE_ALIASES = {
    "fineweb_edu": ("fineweb_edu", "fineweb-edu", "fineweb"),
    "dclm": ("dclm",),
    "finemath_4plus": ("finemath_4plus", "finemath-4plus", "finemath"),
    "cosmopedia_v2": ("cosmopedia_v2", "cosmopedia-v2", "cosmopedia"),
}
DEFAULT_SOURCE_WEIGHTS = {
    "fineweb_edu": 35,
    "dclm": 15,
    "finemath_4plus": 20,
    "cosmopedia_v2": 10,
}
RECORD_SUFFIXES = (
    ".jsonl", ".jsonl.gz", ".jsonl.zst", ".json", ".parquet", ".txt", ".md", ".markdown"
)


def _files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return [p for p in path.rglob("*") if p.is_file() and p.name.lower().endswith(RECORD_SUFFIXES)]


def discover_sources() -> dict[str, Path]:
    out: dict[str, Path] = {}
    previous = ROOT / "runs/frontier-vnext/results/local-data-discovery.json"
    if previous.is_file():
        try:
            for name, value in json.loads(previous.read_text()).items():
                p = Path(value)
                if p.exists() and _files(p):
                    out[name] = p
        except Exception:
            pass
    data = ROOT / "data"
    if not data.is_dir():
        return out
    dirs = [p for p in data.rglob("*") if p.is_dir()]
    for name, aliases in SOURCE_ALIASES.items():
        if name in out:
            continue
        candidates: list[tuple[int, Path]] = []
        for p in dirs:
            low = str(p).lower().replace("-", "_")
            if not any(a.replace("-", "_") in low for a in aliases):
                continue
            fs = _files(p)
            if not fs:
                continue
            score = 0
            if "clean" in low:
                score += 50
            if name.replace("-", "_") in low:
                score += 25
            score -= len(p.parts)
            candidates.append((score, p))
        if candidates:
            candidates.sort(key=lambda x: (x[0], str(x[1])), reverse=True)
            out[name] = candidates[0][1]
    return out


def _record_text(record) -> str | None:
    if isinstance(record, str):
        return record
    if not isinstance(record, dict):
        return None
    for key in (
        "text", "content", "document", "code", "solution", "generated_solution",
        "deepseek_solution", "response",
    ):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def iter_file(path: Path) -> Iterator[str]:
    low = path.name.lower()
    if low.endswith(".parquet"):
        pf = pq.ParquetFile(path)
        names = pf.schema.names
        text_col = next((x for x in ("text", "content", "document", "code", "solution", "response") if x in names), None)
        if text_col is None:
            return
        for batch in pf.iter_batches(batch_size=256, columns=[text_col]):
            for value in batch.column(0).to_pylist():
                if isinstance(value, str):
                    yield value
        return
    if low.endswith(".jsonl.gz"):
        ctx = gzip.open(path, "rt", encoding="utf-8", errors="replace")
    elif low.endswith(".jsonl.zst"):
        import zstandard as zstd
        raw = path.open("rb")
        ctx = io.TextIOWrapper(zstd.ZstdDecompressor().stream_reader(raw), encoding="utf-8", errors="replace")
    elif low.endswith(".jsonl"):
        ctx = path.open("r", encoding="utf-8", errors="replace")
    elif low.endswith(".json"):
        try:
            obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            return
        rows = obj if isinstance(obj, list) else [obj]
        for row in rows:
            text = _record_text(row)
            if text:
                yield text
        return
    else:
        try:
            yield path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            pass
        return
    with ctx as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                text = _record_text(json.loads(line))
            except Exception:
                text = line
            if text:
                yield text


def iter_source(path: Path, seed: int) -> Iterator[str]:
    fs = _files(path)
    rng = random.Random(seed)
    rng.shuffle(fs)
    for file in fs:
        try:
            yield from iter_file(file)
        except Exception as exc:
            print(f"WARNING: skipping {file}: {type(exc).__name__}: {exc}")


def normalize_text(text: str) -> str | None:
    text = text.strip()
    if len(text) < 128:
        return None
    if len(text) > 200_000:
        text = text[:200_000]
    if "\x00" in text:
        return None
    visible = sum(not c.isspace() for c in text)
    if visible < 64:
        return None
    return text


def digest(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8", errors="ignore"), digest_size=16).hexdigest()


def build_tokenizer_sample(
    sources: dict[str, Path], target_bytes: int, seed: int, output_dir: Path
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    sample = output_dir / "tokenizer_sample.txt"
    if sample.is_file() and sample.stat().st_size >= int(target_bytes * 0.9):
        return sample
    iterators = {n: iter_source(p, seed + 1009 * i) for i, (n, p) in enumerate(sorted(sources.items()))}
    names = list(iterators)
    written = 0
    exhausted: set[str] = set()
    with sample.open("w", encoding="utf-8") as out:
        idx = 0
        while written < target_bytes and len(exhausted) < len(names):
            name = names[idx % len(names)]
            idx += 1
            if name in exhausted:
                continue
            try:
                text = normalize_text(next(iterators[name]))
            except StopIteration:
                exhausted.add(name)
                continue
            if not text:
                continue
            payload = text + "\n<|endoftext|>\n"
            out.write(payload)
            written += len(payload.encode("utf-8", errors="ignore"))
    return sample


def train_proxy_tokenizer(sample: Path, vocab_size: int, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file():
        try:
            tok = AsterTokenizer(output)
            if tok.vocab_size == vocab_size:
                return output
        except Exception:
            pass
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        show_progress=True,
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train([str(sample)], trainer=trainer)
    tokenizer.save(str(output))
    checked = AsterTokenizer(output)
    if checked.vocab_size != vocab_size:
        raise RuntimeError(f"Proxy tokenizer produced vocab={checked.vocab_size}, expected {vocab_size}")
    return output


def materialize_proxy(
    sources: dict[str, Path],
    tokenizer_path: Path,
    train_target: int,
    val_target: int,
    seed: int,
    output_dir: Path,
    source_weights: dict[str, int],
) -> tuple[Path, Path, dict]:
    """Materialize a hash-disjoint proxy corpus with *token* quotas per source.

    Document-level weighted sampling is not enough because corpus families have very
    different document-length distributions. This builder therefore computes explicit
    train/validation token targets for every available source, accepts a document only
    while that source/split remains behind quota, and records overshoot. Architecture
    comparisons then see essentially the same token mixture instead of a doc-length
    confound.
    """
    train_file = output_dir / "train.jsonl"
    val_file = output_dir / "val.jsonl"
    meta_file = output_dir / "manifest.json"
    names = [name for name in source_weights if name in sources]
    if len(names) < 2:
        raise RuntimeError(f"Need at least two local corpus families, found {names}")
    total_weight = sum(source_weights[name] for name in names)
    normalized = {name: source_weights[name] / total_weight for name in names}
    if meta_file.is_file() and train_file.is_file() and val_file.is_file():
        try:
            meta = json.loads(meta_file.read_text())
            if (
                meta.get("train_tokens", 0) >= train_target
                and meta.get("val_tokens", 0) >= val_target
                and meta.get("quota_mode") == "per_source_tokens_v2"
                and meta.get("source_weights_normalized") == normalized
            ):
                return train_file, val_file, meta
        except Exception:
            pass

    tok = AsterTokenizer(tokenizer_path)
    def quotas(total: int) -> dict[str, int]:
        q = {n: int(total * normalized[n]) for n in names}
        # Assign integer-rounding remainder deterministically to the largest weights.
        remainder = total - sum(q.values())
        order = sorted(names, key=lambda n: (-normalized[n], n))
        for i in range(remainder):
            q[order[i % len(order)]] += 1
        return q

    train_quota = quotas(train_target)
    val_quota = quotas(val_target)
    its = {n: iter_source(sources[n], seed + 7919 * (i + 1)) for i, n in enumerate(names)}
    seen: set[str] = set()
    counts = {"train": 0, "val": 0}
    docs = defaultdict(int)
    source_tokens = defaultdict(int)
    source_docs = defaultdict(int)
    restarts = defaultdict(int)
    attempts = 0
    max_attempts = 10_000_000

    def source_done(name: str, split: str) -> bool:
        target = train_quota[name] if split == "train" else val_quota[name]
        return source_tokens[f"{split}:{name}"] >= target

    def all_done() -> bool:
        return all(source_done(n, "train") and source_done(n, "val") for n in names)

    with train_file.open("w", encoding="utf-8") as tr, val_file.open("w", encoding="utf-8") as va:
        # Round-robin over source families; quotas, not document count, determine the
        # final mixture. This keeps a giant code/math document from crowding out many
        # shorter educational documents (or vice versa).
        source_index = 0
        while not all_done():
            attempts += 1
            if attempts > max_attempts:
                pending = {
                    f"{split}:{n}": (
                        (train_quota[n] if split == "train" else val_quota[n])
                        - source_tokens[f"{split}:{n}"]
                    )
                    for split in ("train", "val")
                    for n in names
                    if not source_done(n, split)
                }
                raise RuntimeError(f"Proxy quotas not reached after {max_attempts:,} attempts: {pending}")

            name = names[source_index % len(names)]
            source_index += 1
            if source_done(name, "train") and source_done(name, "val"):
                continue
            try:
                raw = next(its[name])
            except StopIteration:
                restarts[name] += 1
                if restarts[name] > 4:
                    raise RuntimeError(
                        f"Source {name!r} exhausted repeatedly before its token quotas were met"
                    )
                its[name] = iter_source(sources[name], seed + 1_000_003 * (restarts[name] + 1))
                continue
            text = normalize_text(raw)
            if not text:
                continue
            h = digest(text)
            if h in seen:
                continue
            seen.add(h)
            split = "val" if int(h[:8], 16) % 10 == 0 else "train"
            if source_done(name, split):
                continue
            token_count = len(tok.encode(text)) + 1
            handle = va if split == "val" else tr
            handle.write(json.dumps({"text": text, "source": name, "hash": h}, ensure_ascii=False) + "\n")
            counts[split] += token_count
            docs[split] += 1
            source_tokens[f"{split}:{name}"] += token_count
            source_docs[f"{split}:{name}"] += 1
            if (docs["train"] + docs["val"]) % 500 == 0:
                print(
                    f"proxy corpus: train={counts['train']:,}/{train_target:,} "
                    f"val={counts['val']:,}/{val_target:,}"
                )

    quota_report = {}
    for split, target_map in (("train", train_quota), ("val", val_quota)):
        for name in names:
            actual = source_tokens[f"{split}:{name}"]
            target = target_map[name]
            quota_report[f"{split}:{name}"] = {
                "target_tokens": target,
                "actual_tokens": actual,
                "overshoot_tokens": actual - target,
                "actual_fraction": actual / max(1, counts[split]),
                "target_fraction": normalized[name],
                "documents": source_docs[f"{split}:{name}"],
            }

    meta = {
        "seed": seed,
        "tokenizer": str(tokenizer_path),
        "vocab_size": tok.vocab_size,
        "train_tokens": counts["train"],
        "val_tokens": counts["val"],
        "train_docs": docs["train"],
        "val_docs": docs["val"],
        "source_tokens": dict(source_tokens),
        "source_quotas": quota_report,
        "source_weights_normalized": normalized,
        "excluded_sources": sorted(set(DEFAULT_SOURCE_WEIGHTS) - set(source_weights)),
        "sources": {k: str(v) for k, v in sources.items()},
        "dedup_hashes": len(seen),
        "split_rule": "blake2b128(text) first32bits mod10 == 0 => validation",
        "quota_mode": "per_source_tokens_v2",
        "note": "Proxy tokenizer/corpus are for controlled architecture experiments, not the final pretraining corpus.",
    }
    meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return train_file, val_file, meta


def _portable_repo_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def write_data_config(train: Path, val: Path, output_dir: Path) -> Path:
    out = output_dir / "data-proxy.yaml"
    payload = {
        "data": {
            "seed": 1337,
            "shuffle_buffer": 10000,
            "min_chars": 64,
            "max_chars": 200000,
            "quality_filters": True,
            "add_eos_between_documents": True,
            "mask_cross_document_loss": True,
            "manifest_path": _portable_repo_path(output_dir / "manifest.json"),
            "sources": [
                {"path": _portable_repo_path(train), "text_field": "text", "weight": 1.0}
            ],
            "validation_sources": [
                {"path": _portable_repo_path(val), "text_field": "text", "weight": 1.0}
            ],
        }
    }
    out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--tokenizer-sample-mib", type=int, default=64)
    parser.add_argument("--train-tokens", type=int, default=20_000_000)
    parser.add_argument("--val-tokens", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--tokenizer-output", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument(
        "--exclude-source",
        action="append",
        default=[],
        choices=sorted(SOURCE_ALIASES),
        help="Exclude a corpus family from tokenizer and proxy materialization; repeatable.",
    )
    args = parser.parse_args()
    output_dir = args.output.resolve()
    tokenizer_output = args.tokenizer_output.resolve()
    discovered = discover_sources()
    excluded = set(args.exclude_source)
    sources = {name: path for name, path in discovered.items() if name not in excluded}
    source_weights = {
        name: weight for name, weight in DEFAULT_SOURCE_WEIGHTS.items() if name not in excluded
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "source-discovery.json").write_text(
        json.dumps({k: str(v) for k, v in sources.items()}, indent=2), encoding="utf-8"
    )
    print("proxy sources:", {k: str(v) for k, v in sources.items()})
    sample = build_tokenizer_sample(
        sources, args.tokenizer_sample_mib * 2**20, args.seed, output_dir
    )
    tokenizer = train_proxy_tokenizer(sample, args.vocab_size, tokenizer_output)
    train, val, meta = materialize_proxy(
        sources,
        tokenizer,
        args.train_tokens,
        args.val_tokens,
        args.seed,
        output_dir,
        source_weights,
    )
    data = write_data_config(train, val, output_dir)
    result = {
        "status": "ok",
        "tokenizer": str(tokenizer),
        "data_config": str(data),
        "manifest": meta,
    }
    (output_dir / "prepare_result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
