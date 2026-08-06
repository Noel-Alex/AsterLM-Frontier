#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import signal
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Stage:
    id: str
    profile: str
    command: list[str]


VALIDATION_CONFIGS: dict[str, list[str]] = {
    "pilot": ["configs/corpus/corpus_pilot_500m.yaml"],
    "frontier": ["configs/corpus/corpus_frontier_16b.yaml", "configs/corpus/stack_edu_2p4b.yaml"],
    "posttrain": ["configs/corpus/posttrain_frontier.yaml"],
    "reasoning": ["configs/corpus/reasoning_frontier.yaml"],
    "benchmarks": ["configs/corpus/decontamination_benchmarks.yaml"],
}


def corpus_stage(profile: str, config: str, source: str) -> Stage:
    return Stage(
        id=f"{profile}-corpus-{source}",
        profile=profile,
        command=[sys.executable, "scripts/materialize_corpus.py", "--config", config, "--only", source],
    )


PROFILES: dict[str, list[Stage]] = {
    # Pilot writes into the same source directories as the frontier corpus. Running
    # frontier later expands those exact checkpoints instead of downloading a duplicate.
    "pilot": [
        corpus_stage("pilot", "configs/corpus/corpus_pilot_500m.yaml", "fineweb_edu"),
        corpus_stage("pilot", "configs/corpus/corpus_pilot_500m.yaml", "cosmopedia_v2"),
        corpus_stage("pilot", "configs/corpus/corpus_pilot_500m.yaml", "finemath_4plus"),
        corpus_stage("pilot", "configs/corpus/corpus_pilot_500m.yaml", "dclm"),
    ],
    "frontier": [
        corpus_stage("frontier", "configs/corpus/corpus_frontier_16b.yaml", "fineweb_edu"),
        corpus_stage("frontier", "configs/corpus/corpus_frontier_16b.yaml", "dclm"),
        corpus_stage("frontier", "configs/corpus/corpus_frontier_16b.yaml", "cosmopedia_v2"),
        corpus_stage("frontier", "configs/corpus/corpus_frontier_16b.yaml", "finemath_4plus"),
        Stage(
            id="frontier-stack-edu",
            profile="frontier",
            command=[
                sys.executable,
                "scripts/prepare_stack_edu_multilang.py",
                "--config",
                "configs/corpus/stack_edu_2p4b.yaml",
            ],
        ),
    ],
    "posttrain": [
        Stage(
            id="posttrain-records",
            profile="posttrain",
            command=[
                sys.executable,
                "scripts/materialize_hf_records.py",
                "--config",
                "configs/corpus/posttrain_frontier.yaml",
            ],
        )
    ],

    "reasoning": [
        Stage(
            id=f"reasoning-{source}",
            profile="reasoning",
            command=[
                sys.executable,
                "scripts/materialize_hf_records.py",
                "--config",
                "configs/corpus/reasoning_frontier.yaml",
                "--only",
                source,
            ],
        )
        for source in ("dapo_math", "verifiable_python", "mixture_of_thoughts")
    ],
    "benchmarks": [
        Stage(
            id="decontamination-benchmarks",
            profile="benchmarks",
            command=[
                sys.executable,
                "scripts/materialize_hf_records.py",
                "--config",
                "configs/corpus/decontamination_benchmarks.yaml",
            ],
        )
    ],
}


NETWORK_MODES: dict[str, dict[str, str]] = {
    "low": {
        "HF_HUB_DOWNLOAD_TIMEOUT": "600",
        "HF_HUB_ETAG_TIMEOUT": "120",
        "HF_XET_NUM_CONCURRENT_RANGE_GETS": "2",
    },
    "balanced": {
        "HF_HUB_DOWNLOAD_TIMEOUT": "300",
        "HF_HUB_ETAG_TIMEOUT": "60",
        "HF_XET_NUM_CONCURRENT_RANGE_GETS": "8",
    },
    "fast": {
        "HF_HUB_DOWNLOAD_TIMEOUT": "120",
        "HF_HUB_ETAG_TIMEOUT": "30",
        "HF_XET_NUM_CONCURRENT_RANGE_GETS": "16",
        "HF_XET_HIGH_PERFORMANCE": "1",
    },
}


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def existing_hf_token_path(env: dict[str, str]) -> Path:
    """Return the token location used before this process relocates HF_HOME.

    Hugging Face stores the login token under HF_HOME by default. AsterLM uses a
    project-local HF_HOME for large caches, so without preserving HF_TOKEN_PATH a
    valid global `hf auth login` is hidden from child processes.
    """
    explicit = env.get("HF_TOKEN_PATH")
    if explicit:
        return Path(explicit).expanduser()
    original_home = env.get("HF_HOME")
    if original_home:
        return Path(original_home).expanduser() / "token"
    xdg_cache = env.get("XDG_CACHE_HOME")
    if xdg_cache:
        return Path(xdg_cache).expanduser() / "huggingface" / "token"
    return Path.home() / ".cache" / "huggingface" / "token"


def execution_context_errors(
    *,
    repo_root: Path,
    cwd: Path,
    virtual_env: str | None,
    allow_external_venv: bool,
) -> list[str]:
    errors: list[str] = []
    if cwd.resolve() != repo_root.resolve():
        errors.append(
            f"Run this command from the repository root: {repo_root} (current directory: {cwd.resolve()})"
        )
    if virtual_env and not allow_external_venv:
        active_repo = Path(virtual_env).expanduser().resolve().parent
        if active_repo != repo_root.resolve():
            errors.append(
                "The active virtual environment belongs to a different checkout: "
                f"{Path(virtual_env).expanduser().resolve()}. Deactivate it and run "
                f"`source {repo_root / '.venv/bin/activate'}`. Use --allow-external-venv "
                "only when sharing that environment is intentional."
            )
    return errors


def build_environment(args: argparse.Namespace) -> dict[str, str]:
    env = dict(os.environ)
    token_path = existing_hf_token_path(env)
    hf_home = Path(args.hf_home).expanduser().resolve()
    hf_home.mkdir(parents=True, exist_ok=True)
    env["HF_HOME"] = str(hf_home)
    env.setdefault("HF_HUB_CACHE", str(hf_home / "hub"))
    env.setdefault("HF_XET_CACHE", str(hf_home / "xet"))
    # Keep authentication independent from the project-local cache. HF_TOKEN has
    # higher priority, and an explicitly supplied HF_TOKEN_PATH is never replaced.
    if not env.get("HF_TOKEN") and not env.get("HF_TOKEN_PATH") and token_path.is_file():
        env["HF_TOKEN_PATH"] = str(token_path.resolve())
    env.update(NETWORK_MODES[args.network_mode])
    env.setdefault("HF_HUB_VERBOSITY", "warning")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.network_mode != "fast":
        env.pop("HF_XET_HIGH_PERFORMANCE", None)
    return env


def execute_stage(
    stage: Stage,
    *,
    env: dict[str, str],
    log_dir: Path,
    dry_run: bool,
    command_retries: int,
    retry_seconds: float,
    interrupt_grace_seconds: float,
) -> dict[str, Any]:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{stage.id}.log"
    result: dict[str, Any] = {
        "id": stage.id,
        "profile": stage.profile,
        "command": stage.command,
        "log": str(log_path),
        "attempts": [],
    }
    print("$", " ".join(stage.command), flush=True)
    if dry_run:
        result["dry_run"] = True
        return result

    max_attempts = max(1, command_retries + 1)
    for attempt in range(1, max_attempts + 1):
        started = time.time()
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n=== attempt {attempt}/{max_attempts} at {time.ctime(started)} ===\n")
            process = subprocess.Popen(
                stage.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            assert process.stdout is not None
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                returncode = process.wait()
            except KeyboardInterrupt:
                # Forward an interrupt rather than terminating immediately. The
                # materializers catch SIGINT/KeyboardInterrupt, commit their latest
                # remote cursor and finalize the current local shard. A second
                # Ctrl-C forces an immediate kill.
                try:
                    if os.name == "posix":
                        process.send_signal(signal.SIGINT)
                    else:
                        process.terminate()
                    try:
                        remaining, _ = process.communicate(timeout=interrupt_grace_seconds)
                        if remaining:
                            print(remaining, end="", flush=True)
                            log.write(remaining)
                            log.flush()
                    except KeyboardInterrupt:
                        process.kill()
                        process.wait(timeout=15)
                        raise
                    except subprocess.TimeoutExpired:
                        process.terminate()
                        try:
                            remaining, _ = process.communicate(timeout=15)
                            if remaining:
                                print(remaining, end="", flush=True)
                                log.write(remaining)
                                log.flush()
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=15)
                finally:
                    raise
        attempt_result = {
            "attempt": attempt,
            "returncode": returncode,
            "started_at_unix": started,
            "seconds": time.time() - started,
        }
        result["attempts"].append(attempt_result)
        if returncode == 0:
            result["returncode"] = 0
            return result
        if attempt < max_attempts:
            delay = min(300.0, retry_seconds * (2 ** (attempt - 1)))
            print(f"Stage {stage.id} exited {returncode}; retrying in {delay:.1f}s", flush=True)
            time.sleep(delay)
    result["returncode"] = result["attempts"][-1]["returncode"]
    return result


def profiles_for(args: argparse.Namespace) -> list[str]:
    if args.profile != "all":
        return [args.profile]
    # Pilot is a progressive first tranche of frontier, not a duplicate output.
    return ["pilot", "benchmarks", "reasoning", "posttrain", "frontier"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Start interruption-safe, low-bandwidth AsterLM data downloads")
    parser.add_argument("--profile", choices=["pilot", "frontier", "posttrain", "reasoning", "benchmarks", "all"], default="pilot")
    parser.add_argument("--network-mode", choices=sorted(NETWORK_MODES), default="low")
    parser.add_argument("--hf-home", default="data/hf-cache")
    parser.add_argument("--allow-external-venv", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-first", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--require-auth", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--command-retries", type=int, default=1)
    parser.add_argument("--command-retry-seconds", type=float, default=30.0)
    parser.add_argument("--interrupt-grace-seconds", type=float, default=300.0)
    parser.add_argument("--max-retries", type=int, default=50, help="Passed to materializers; 0 means unlimited")
    parser.add_argument("--retry-base-seconds", type=float, default=5.0)
    parser.add_argument("--retry-max-seconds", type=float, default=300.0)
    parser.add_argument("--checkpoint-seconds", type=float, default=900.0)
    parser.add_argument("--manifest", default="data/download_run.json")
    parser.add_argument("--log-dir", default="data/download-logs")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    context_errors = execution_context_errors(
        repo_root=repo_root,
        cwd=Path.cwd(),
        virtual_env=os.environ.get("VIRTUAL_ENV"),
        allow_external_venv=args.allow_external_venv,
    )
    if context_errors:
        raise SystemExit("\n".join(context_errors))

    selected_profiles = profiles_for(args)
    stages: list[Stage] = []

    if not args.skip_preflight:
        preflight = [
            sys.executable,
            "scripts/data_preflight.py",
            "--output",
            "data/data_preflight.json",
            "--minimum-free-gib",
            "90" if args.profile == "all" else "5",
        ]
        if args.require_auth:
            preflight.append("--require-auth")
        stages.append(Stage("data-preflight", "preflight", preflight))

    if args.validate_first:
        configs: list[str] = []
        for profile in selected_profiles:
            # Frontier validation supersedes pilot because it has the same web sources
            # plus the full Stack-Edu configuration.
            if profile == "pilot" and "frontier" in selected_profiles:
                continue
            configs.extend(VALIDATION_CONFIGS[profile])
        configs = list(dict.fromkeys(configs))
        stages.append(
            Stage(
                id=f"validate-{args.profile}",
                profile="validation",
                command=[
                    sys.executable,
                    "scripts/validate_data_sources.py",
                    *configs,
                    "--output",
                    f"data/source_validation_{args.profile}.json",
                    "--continue-on-error",
                    "--max-retries",
                    "10",
                ],
            )
        )

    for profile in selected_profiles:
        for stage in PROFILES[profile]:
            command = list(stage.command)
            if stage.id != "data-preflight":
                command.extend(
                    [
                        "--max-retries",
                        str(args.max_retries),
                        "--retry-base-seconds",
                        str(args.retry_base_seconds),
                        "--retry-max-seconds",
                        str(args.retry_max_seconds),
                        "--checkpoint-seconds",
                        str(args.checkpoint_seconds),
                    ]
                )
            stages.append(Stage(stage.id, stage.profile, command))

    usage = shutil.disk_usage(Path.cwd())
    estimates_gib = {"pilot": 4, "frontier": 70, "posttrain": 15, "reasoning": 12, "benchmarks": 2, "all": 105}
    needed = estimates_gib[args.profile]
    print(f"Free disk: {usage.free / 2**30:.1f} GiB; conservative profile estimate: {needed} GiB")
    if not args.dry_run and usage.free / 2**30 < needed:
        raise SystemExit("Not enough free disk for the selected profile")

    env = build_environment(args)
    path = Path(args.manifest)
    log_dir = Path(args.log_dir)
    manifest: dict[str, Any] = {
        "version": 2,
        "profile": args.profile,
        "profiles": selected_profiles,
        "network_mode": args.network_mode,
        "hf_home": env["HF_HOME"],
        "started_at_unix": time.time(),
        "stages": [],
        "free_disk_gib_before": usage.free / 2**30,
        "environment": {
            key: env.get(key)
            for key in (
                "HF_HOME",
                "HF_HUB_CACHE",
                "HF_XET_CACHE",
                "HF_TOKEN_PATH",
                "HF_HUB_DOWNLOAD_TIMEOUT",
                "HF_HUB_ETAG_TIMEOUT",
                "HF_XET_NUM_CONCURRENT_RANGE_GETS",
                "HF_XET_HIGH_PERFORMANCE",
            )
        },
    }
    atomic_write(path, manifest)

    for stage in stages:
        stage_retries = 0 if stage.profile in {"preflight", "validation"} else args.command_retries
        result = execute_stage(
            stage,
            env=env,
            log_dir=log_dir,
            dry_run=args.dry_run,
            command_retries=stage_retries,
            retry_seconds=args.command_retry_seconds,
            interrupt_grace_seconds=args.interrupt_grace_seconds,
        )
        manifest["stages"].append(result)
        manifest["updated_at_unix"] = time.time()
        atomic_write(path, manifest)
        if result.get("returncode") not in (None, 0) and not args.continue_on_error:
            manifest["status"] = "failed"
            manifest["failed_stage"] = stage.id
            atomic_write(path, manifest)
            raise SystemExit(int(result["returncode"]))

    manifest["status"] = "complete"
    manifest["finished_at_unix"] = time.time()
    manifest["free_disk_gib_after"] = shutil.disk_usage(Path.cwd()).free / 2**30
    atomic_write(path, manifest)
    print(f"Download manifest: {path}")
    print(f"Per-stage logs: {log_dir}")


if __name__ == "__main__":
    main()
