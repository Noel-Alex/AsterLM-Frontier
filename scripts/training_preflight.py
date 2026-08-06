#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from asterlm.config import AsterConfig, DataConfig, TrainConfig
from asterlm.data.mixture import _looks_like_local_path, iter_source, local_data_paths
from asterlm.data.tokenizer import AsterTokenizer
from asterlm.training.checkpoint import resolve_checkpoint


@dataclass
class Check:
    level: str
    name: str
    detail: str


def add(checks: list[Check], level: str, name: str, detail: str) -> None:
    checks.append(Check(level, name, detail))


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def inspect_checkpoint(path: str | None, checks: list[Check]) -> None:
    if not path:
        return
    resolved = resolve_checkpoint(path)
    if not resolved.exists():
        add(checks, "ERROR", "checkpoint", f"Missing checkpoint: {resolved}")
        return
    if resolved.is_file():
        if resolved.suffix not in {".pt", ".safetensors"}:
            add(checks, "WARN", "checkpoint", f"Unusual checkpoint file: {resolved}")
        else:
            add(checks, "OK", "checkpoint", f"Found weights file: {resolved}")
        return
    weights = [candidate for candidate in (resolved / "model.safetensors", resolved / "model.pt") if candidate.exists()]
    if not weights:
        add(checks, "ERROR", "checkpoint", f"No model.safetensors or model.pt under {resolved}")
    else:
        add(checks, "OK", "checkpoint", f"Found {weights[0].name} under {resolved}")
    if (resolved / "trainer_state.pt").exists():
        add(checks, "OK", "resume state", "Optimizer, RNG, step, and token state are present")
    else:
        add(checks, "WARN", "resume state", "Weights-only checkpoint; full training resume is unavailable")


def inspect_data(config: DataConfig, checks: list[Check], check_first_record: bool) -> None:
    total_bytes = 0
    for label, sources in (("train", config.sources), ("validation", config.validation_sources)):
        for index, source in enumerate(sources):
            name = f"{label} source {index + 1}: {source.path}"
            path = Path(source.path)
            if not path.exists():
                if _looks_like_local_path(source.path):
                    add(checks, "ERROR", name, "Configured local source does not exist")
                else:
                    add(checks, "WARN", name, "Remote Hugging Face source; training requires network/cache access")
                continue
            files = local_data_paths(path)
            if not files:
                add(checks, "ERROR", name, "No supported records/text shards found")
                continue
            size = sum(item.stat().st_size for item in files)
            total_bytes += size
            add(checks, "OK", name, f"{len(files):,} usable file(s), {size / (1024**2):.1f} MiB")
            partials = list(path.rglob("*.partial")) + list(path.rglob("*.tmp")) if path.is_dir() else []
            if partials:
                add(checks, "WARN", f"{name} partials", f"{len(partials)} incomplete temporary file(s) remain")
            if check_first_record:
                try:
                    record = next(iter_source(source, config.seed + index, 1))
                except StopIteration:
                    add(checks, "ERROR", f"{name} content", "Source yielded zero records")
                except Exception as exc:
                    add(checks, "ERROR", f"{name} content", f"Could not read first record: {exc}")
                else:
                    add(checks, "OK", f"{name} content", f"First record fields: {sorted(record)[:12]}")
    if total_bytes:
        add(checks, "OK", "local data total", f"{total_bytes / (1024**3):.2f} GiB of usable local shards")


def inspect_dependencies(model: AsterConfig, train: TrainConfig, checks: list[Check]) -> None:
    requirements: list[tuple[str, str]] = []
    if train.optimizer in {"apollo", "apollo_mini"}:
        requirements.append(("apollo_torch", "APOLLO optimizer"))
    if train.optimizer.startswith("torchao_") or model.ffn_backend == "loqt_int4":
        requirements.append(("torchao", "torchao optimizer/LoQT path"))
    if train.precision_backend == "transformer_engine_fp8" or model.linear_backend == "transformer_engine":
        requirements.append(("transformer_engine", "Transformer Engine FP8 path"))
    if model.kda_backend == "fla":
        requirements.append(("fla", "FLA KDA backend pinned by config"))
    for module, purpose in requirements:
        if module_available(module):
            add(checks, "OK", purpose, f"Python module {module!r} is importable")
        else:
            add(checks, "ERROR", purpose, f"Missing Python module {module!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate AsterLM configs, data, tokenizer, hardware, and checkpoints "
            "before training"
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--data", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--check-first-record", action="store_true")
    parser.add_argument("--json", dest="json_path", default=None, help="Optional JSON report path")
    parser.add_argument("--allow-warnings", action="store_true", help="Return success when only warnings remain")
    args = parser.parse_args()

    checks: list[Check] = []
    try:
        model = AsterConfig.from_yaml(args.model)
        train = TrainConfig.from_yaml(args.train)
    except Exception as exc:
        print(f"ERROR config parse: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    add(checks, "OK", "model config", f"Loaded {args.model}")
    add(checks, "OK", "train config", f"Loaded {args.train}")
    if train.sequence_length > model.max_seq_len:
        add(
            checks,
            "ERROR",
            "sequence length",
            f"Train length {train.sequence_length:,} exceeds model max {model.max_seq_len:,}",
        )
    else:
        add(checks, "OK", "sequence length", f"{train.sequence_length:,} <= {model.max_seq_len:,}")

    tokenizer_path = Path(train.tokenizer_path)
    if not tokenizer_path.exists():
        add(checks, "ERROR", "tokenizer", f"Missing {tokenizer_path}")
    else:
        try:
            tokenizer = AsterTokenizer(tokenizer_path)
        except Exception as exc:
            add(checks, "ERROR", "tokenizer", str(exc))
        else:
            if tokenizer.vocab_size != model.vocab_size:
                add(
                    checks,
                    "ERROR",
                    "tokenizer vocabulary",
                    f"Tokenizer has {tokenizer.vocab_size:,}; model expects {model.vocab_size:,}",
                )
            else:
                add(checks, "OK", "tokenizer vocabulary", f"Tokenizer and model both use {model.vocab_size:,} tokens")

    if args.data:
        try:
            data = DataConfig.from_yaml(args.data)
        except Exception as exc:
            add(checks, "ERROR", "data config", f"Could not load {args.data}: {exc}")
        else:
            add(checks, "OK", "data config", f"Loaded {args.data} with {len(data.sources)} train source(s)")
            if not data.sources:
                add(checks, "ERROR", "training sources", "Data config has no training sources")
            inspect_data(data, checks, args.check_first_record)

    inspect_checkpoint(args.checkpoint or train.resume, checks)
    inspect_dependencies(model, train, checks)

    requested_cuda = train.device.startswith("cuda")
    if requested_cuda and not torch.cuda.is_available():
        add(checks, "ERROR", "CUDA", "Training requests CUDA, but torch.cuda.is_available() is false")
    elif requested_cuda:
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        add(checks, "OK", "CUDA", f"{name}, {total:.1f} GiB")
        if train.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            add(checks, "ERROR", "BF16", "GPU/PyTorch reports BF16 is unsupported")
        else:
            add(checks, "OK", "BF16", "BF16 training is supported")

    output = Path(train.output_dir)
    if output.exists() and any(output.iterdir()) and not (args.checkpoint or train.resume):
        add(checks, "WARN", "output directory", f"{output} is non-empty; a fresh run may mix artifacts")
    else:
        add(checks, "OK", "output directory", str(output))
    free = shutil.disk_usage(Path.cwd()).free / (1024**3)
    add(checks, "OK" if free >= 20 else "WARN", "free disk", f"{free:.1f} GiB available on repository filesystem")

    order = {"ERROR": 0, "WARN": 1, "OK": 2}
    for check in sorted(checks, key=lambda item: order[item.level]):
        print(f"[{check.level:5}] {check.name}: {check.detail}")
    errors = sum(check.level == "ERROR" for check in checks)
    warnings = sum(check.level == "WARN" for check in checks)
    passed = len(checks) - errors - warnings
    print(
        f"\nPreflight result: {errors} error(s), {warnings} warning(s), "
        f"{passed} passed check(s)"
    )

    if args.json_path:
        report = {
            "model": args.model,
            "train": args.train,
            "data": args.data,
            "checkpoint": args.checkpoint or train.resume,
            "checks": [asdict(check) for check in checks],
            "errors": errors,
            "warnings": warnings,
        }
        path = Path(args.json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if errors or (warnings and not args.allow_warnings):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
