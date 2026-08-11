#!/usr/bin/env python
from __future__ import annotations

import argparse
import difflib
import json
import multiprocessing as mp
import queue
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from asterlm.data.resumable import RetryPolicy, atomic_write_json, is_retryable_exception


@dataclass(slots=True)
class SourceSpec:
    id: str
    path: str
    name: str | None
    split: str
    required_fields: tuple[str, ...]
    revision: str | None = None
    columns: tuple[str, ...] = ()


def nested_get(record: dict[str, Any], field: str) -> Any:
    value: Any = record
    for part in field.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def specs_from_config(path: Path) -> list[SourceSpec]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    specs: list[SourceSpec] = []
    if "corpus" in raw:
        for item in raw["corpus"]["sources"]:
            specs.append(
                SourceSpec(
                    id=item["id"],
                    path=item["path"],
                    name=item.get("name"),
                    split=item.get("split", "train"),
                    required_fields=(item.get("text_field", "text"),),
                    revision=item.get("revision"),
                    columns=tuple(item.get("columns") or ()),
                )
            )
    elif "records" in raw:
        for item in raw["records"]["sources"]:
            fields = tuple(item.get("keep_fields") or ())
            specs.append(
                SourceSpec(
                    id=item["id"],
                    path=item["path"],
                    name=item.get("name"),
                    split=item["split"],
                    required_fields=fields,
                    revision=item.get("revision"),
                    columns=tuple(item.get("columns") or ()),
                )
            )
    elif "stack_edu" in raw:
        for item in raw["stack_edu"]["languages"]:
            specs.append(
                SourceSpec(
                    id=f"stack_edu_{item['name']}",
                    path="HuggingFaceTB/stack-edu",
                    name=item["name"],
                    split="train",
                    required_fields=("blob_id", "license_type"),
                )
            )
    else:
        raise ValueError(f"{path} is not a corpus, record, or Stack-Edu configuration")
    return specs


def closest(target: str, options: Iterable[str]) -> list[str]:
    return difflib.get_close_matches(target, list(options), n=5, cutoff=0.25)


def validate_once(spec: SourceSpec, sample: bool) -> dict[str, Any]:
    # fsspec registers optional codecs when imported after zstandard is present.
    import zstandard  # noqa: F401
    from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset
    from huggingface_hub import HfApi

    result: dict[str, Any] = {
        "id": spec.id,
        "path": spec.path,
        "name": spec.name,
        "split": spec.split,
        "status": "started",
    }
    info = HfApi().dataset_info(spec.path, revision=spec.revision)
    revision = info.sha
    result["resolved_revision"] = revision
    result["license"] = (info.card_data or {}).get("license") if info.card_data else None
    configs = get_dataset_config_names(spec.path, revision=revision)
    result["configs"] = configs
    selected_name = spec.name
    if selected_name is not None and selected_name not in configs:
        result["status"] = "invalid_config"
        result["suggestions"] = closest(selected_name, configs)
        return result
    splits = get_dataset_split_names(spec.path, config_name=selected_name, revision=revision)
    result["splits"] = splits
    if spec.split not in splits:
        result["status"] = "invalid_split"
        result["suggestions"] = closest(spec.split, splits)
        return result
    if sample:
        kwargs: dict[str, Any] = {
            "path": spec.path,
            "name": selected_name,
            "split": spec.split,
            "streaming": True,
            "revision": revision,
        }
        if spec.columns:
            kwargs["columns"] = list(spec.columns)
        dataset = load_dataset(**kwargs)
        record = next(iter(dataset))
        missing = [field for field in spec.required_fields if nested_get(record, field) is None]
        result["sample_fields"] = sorted(record.keys()) if isinstance(record, dict) else []
        result["missing_required_fields"] = missing
        if missing:
            result["status"] = "missing_fields"
            return result
        if hasattr(dataset, "state_dict"):
            state = dataset.state_dict()
            result["shard_cursor_supported"] = state is not None
        else:
            result["shard_cursor_supported"] = False
    result["status"] = "ok"
    return result


def validate(spec: SourceSpec, sample: bool, policy: RetryPolicy) -> dict[str, Any]:
    failures = 0
    while True:
        try:
            return validate_once(spec, sample)
        except Exception as exc:  # noqa: BLE001 - classify arbitrary network/backend failures
            failures += 1
            if not is_retryable_exception(exc) or not policy.permits(failures):
                return {
                    "id": spec.id,
                    "path": spec.path,
                    "name": spec.name,
                    "split": spec.split,
                    "status": "error",
                    "attempts": failures,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            delay = policy.delay(failures)
            print(
                f"{spec.id}: transient validation failure {failures}: "
                f"{type(exc).__name__}: {exc}; retrying in {delay:.1f}s",
                flush=True,
            )
            time.sleep(delay)


def _validate_worker(
    result_queue: Any,
    spec: SourceSpec,
    sample: bool,
    policy: RetryPolicy,
) -> None:
    """Run one source probe out of process so native loader hangs are killable."""
    try:
        result_queue.put(validate(spec, sample, policy))
    except Exception as exc:  # noqa: BLE001 - serialize child failures for the parent
        result_queue.put(
            {
                "id": spec.id,
                "path": spec.path,
                "name": spec.name,
                "split": spec.split,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )


def validate_with_timeout(
    spec: SourceSpec,
    sample: bool,
    policy: RetryPolicy,
    timeout_seconds: float,
) -> dict[str, Any]:
    if timeout_seconds <= 0:
        return validate(spec, sample, policy)

    context = mp.get_context("spawn")
    result_queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_validate_worker,
        args=(result_queue, spec, sample, policy),
        daemon=True,
    )
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(10)
        if process.is_alive():
            process.kill()
            process.join(10)
        result_queue.cancel_join_thread()
        result_queue.close()
        return {
            "id": spec.id,
            "path": spec.path,
            "name": spec.name,
            "split": spec.split,
            "status": "timeout",
            "timeout_seconds": timeout_seconds,
            "error": "source probe exceeded its wall-clock deadline",
        }

    try:
        return result_queue.get(timeout=5)
    except queue.Empty:
        return {
            "id": spec.id,
            "path": spec.path,
            "name": spec.name,
            "split": spec.split,
            "status": "error",
            "error": f"source probe exited with code {process.exitcode} without a result",
        }
    finally:
        result_queue.cancel_join_thread()
        result_queue.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate Hugging Face dataset configs/splits and optionally sample required fields"
    )
    parser.add_argument("config", nargs="+")
    parser.add_argument("--no-sample", action="store_true", help="Only validate metadata; do not stream one row")
    parser.add_argument("--output", default="data/source_validation.json")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--source-id",
        action="append",
        default=[],
        help="Validate only the named source id; repeat to select multiple sources",
    )
    parser.add_argument("--max-retries", type=int, default=10, help="0 means unlimited transient retries")
    parser.add_argument("--retry-base-seconds", type=float, default=3.0)
    parser.add_argument("--retry-max-seconds", type=float, default=60.0)
    parser.add_argument(
        "--source-timeout-seconds",
        type=float,
        default=300.0,
        help="Kill and record a source probe that exceeds this deadline; 0 disables the deadline",
    )
    args = parser.parse_args()

    policy = RetryPolicy(args.max_retries, args.retry_base_seconds, args.retry_max_seconds)
    results: list[dict[str, Any]] = []
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    for config in args.config:
        path = Path(config)
        for spec in specs_from_config(path):
            if args.source_id and spec.id not in args.source_id:
                continue
            print(f"validating {spec.id}: {spec.path}/{spec.name or '<default>'}:{spec.split}", flush=True)
            result = validate_with_timeout(
                spec,
                sample=not args.no_sample,
                policy=policy,
                timeout_seconds=args.source_timeout_seconds,
            )
            result["config_file"] = str(path)
            results.append(result)
            print(json.dumps(result, indent=2, default=str))
            atomic_write_json(output, {"results": results, "updated_at_unix": time.time()})
            if result["status"] != "ok" and not args.continue_on_error:
                raise SystemExit(2)

    failures = [row for row in results if row["status"] != "ok"]
    atomic_write_json(
        output,
        {
            "results": results,
            "validated_sources": len(results),
            "failures": len(failures),
            "finished_at_unix": time.time(),
        },
    )
    print(f"validated {len(results)} sources; failures={len(failures)}; report={output}")
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
