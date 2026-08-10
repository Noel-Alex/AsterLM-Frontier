#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'data' / 'aster-studio'

PROFILES: dict[str, list[str]] = {
    "kda": ["flash-linear-attention==0.5.2"],
    "apollo": ["apollo-torch>=1.0"],
    "tracking": ["wandb>=0.19", "tensorboard>=2.18"],
    "torchao": ["torchao==0.17.0"],
    "fp8": ["transformer-engine[pytorch]==2.17.0"],
    "reasoning": ["math-verify>=0.7.0", "sympy>=1.13"],
}



def run(
    cmd: list[str],
    *,
    capture: bool = False,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    print('$', ' '.join(cmd), flush=True)
    return subprocess.run(
        cmd,
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        env=env,
    )


def installer_prefix() -> list[str]:
    candidates = [
        os.environ.get("ASTERLM_UV_PATH"),
        shutil.which("uv"),
        "/root/.local/bin/uv",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return [candidate, "pip"]
    try:
        import pip  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Neither uv nor pip is available in this environment") from exc
    return [sys.executable, "-m", "pip"]


def pip_freeze(path: Path) -> None:
    result = run([*installer_prefix(), "freeze"], capture=True)
    path.write_text(result.stdout or '', encoding='utf-8')


def dry_run_guard(requirements: list[str]) -> dict[str, Any]:
    prefix = installer_prefix()
    if prefix[-1] == "pip" and Path(prefix[0]).name.startswith("uv"):
        result = subprocess.run(
            [*prefix, "install", "--upgrade", "--dry-run", *requirements],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        print(result.stdout or "", flush=True)
        if result.returncode != 0:
            raise RuntimeError("uv dependency dry-run failed")
        protected = {"torch", "triton", "datasets", "pyarrow", "huggingface-hub", "zstandard"}
        planned_changes: dict[str, str | None] = {}
        for line in (result.stdout or "").splitlines():
            match = re.match(r"\s*[+\-~]\s+([A-Za-z0-9_.-]+)==([^\s]+)", line)
            if match and match.group(1).lower().replace("_", "-") in protected:
                planned_changes[match.group(1)] = match.group(2)
        if planned_changes:
            raise RuntimeError(
                "Refusing optional runtime install because uv wants to replace protected "
                f"packages: {planned_changes}"
            )
        return {"installer": "uv", "output": result.stdout or ""}

    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as handle:
        report_path = Path(handle.name)
    try:
        result = subprocess.run(
            [
                *prefix, 'install', '--upgrade',
                '--dry-run', '--report', str(report_path), *requirements,
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        print(result.stdout or '', flush=True)
        if result.returncode != 0:
            raise RuntimeError('pip dependency dry-run failed')
        report = json.loads(report_path.read_text(encoding='utf-8'))
    finally:
        report_path.unlink(missing_ok=True)

    protected = {"torch", "triton", "datasets", "pyarrow", "huggingface-hub", "zstandard"}
    planned_changes: dict[str, str | None] = {}
    for item in report.get("install", []):
        metadata = item.get("metadata") or {}
        name = str(metadata.get("name", "")).lower().replace("_", "-")
        if name in protected:
            planned_changes[name] = metadata.get("version")
    if planned_changes:
        raise RuntimeError(
            "Refusing optional runtime install because pip wants to replace protected "
            f"packages: {planned_changes}. The current CUDA/downloader environment is "
            "working; resolve this profile separately instead of letting pip mutate it."
        )
    return report


def verify() -> dict[str, Any]:
    import torch
    checks: dict[str, Any] = {
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    probes = {
        'fla_kda': ('fla.layers.kda', 'KimiDeltaAttention'),
        'apollo': ('apollo_torch', 'APOLLOAdamW'),
        'torchao': ('torchao.optim', 'AdamW4bit'),
        'transformer_engine': ('transformer_engine.pytorch', None),
        'wandb': ('wandb', None),
        'tensorboard': ('tensorboard', None),
    }
    for name, (module, attr) in probes.items():
        try:
            loaded = importlib.import_module(module)
            if attr is not None:
                getattr(loaded, attr)
            checks[name] = True
        except Exception as exc:  # noqa: BLE001 - every optional capability must be reported
            checks[name] = f'{type(exc).__name__}: {exc}'
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Install AsterLM optional runtime dependencies without replacing the working PyTorch build.'
    )
    parser.add_argument("profile", choices=["kda", "apollo", "tracking", "torchao", "fp8", "reasoning", "baseline", "all"])
    args = parser.parse_args()

    STATE.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime('%Y%m%d-%H%M%S')
    before = STATE / f'pip-freeze-before-runtime-{stamp}.txt'
    after = STATE / f'pip-freeze-after-runtime-{stamp}.txt'
    pip_freeze(before)

    if args.profile == "baseline":
        profiles = ["kda", "apollo", "tracking"]
    elif args.profile == "all":
        profiles = ["kda", "apollo", "tracking", "torchao", "reasoning"]
    else:
        profiles = [args.profile]
    import torch
    torch_before = torch.__version__
    summary: dict[str, Any] = {'torch_before': torch_before, 'profiles': profiles, 'steps': []}

    for profile in profiles:
        requirements = PROFILES[profile]
        print(f'\n=== {profile.upper()} ===', flush=True)
        if profile == "fp8":
            if os.name == "nt":
                raise RuntimeError("Install FP8 inside WSL/Linux, not native Windows Python")
            helper = ROOT / "scripts" / "install_transformer_engine_linux.sh"
            helper_env = dict(os.environ)
            helper_env["ASTERLM_VENV_PATH"] = sys.prefix
            run(["bash", str(helper)], env=helper_env)
            summary['steps'].append(
                {'profile': profile, 'requirements': requirements, 'helper': str(helper), 'ok': True}
            )
            continue
        dry_run_guard(requirements)
        install_cmd = [*installer_prefix(), "install", "--upgrade"]
        install_cmd.extend(requirements)
        run(install_cmd)
        import torch as torch_after_step
        if torch_after_step.__version__ != torch_before:
            raise RuntimeError(
                f'PyTorch changed unexpectedly: {torch_before} -> {torch_after_step.__version__}'
            )
        summary['steps'].append({'profile': profile, 'requirements': requirements, 'ok': True})

    pip_freeze(after)
    summary['verification'] = verify()
    summary['before_freeze'] = str(before.relative_to(ROOT))
    summary['after_freeze'] = str(after.relative_to(ROOT))
    output = STATE / 'runtime-setup-latest.json'
    output.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('\n' + json.dumps(summary, indent=2), flush=True)
    print('\nRuntime installation complete. Re-run the Studio capability audit next.', flush=True)


if __name__ == '__main__':
    main()
