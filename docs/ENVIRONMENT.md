# Environment guide: RTX 4080 Laptop GPU

## Recommended platform

1. Ubuntu 24.04 under WSL2 or native Linux.
2. A stable existing Linux installation with NVIDIA CUDA working.
3. Native Windows only for editing/reference execution; research CUDA packages are less reliable there.

## Installation

```bash
PYTHON_BIN=python3.12 bash scripts/setup_linux.sh \
  --with-apollo --with-torchao --with-tracking
source .venv/bin/activate
```

The setup script creates or reuses `.venv`. Add `--skip-torch` when that
environment already contains a known-good CUDA PyTorch installation.

Optional FP8:

```bash
PYTHON_BIN=python3.12 bash scripts/setup_linux.sh \
  --with-apollo --with-fp8 --with-torchao --with-tracking
```

Do not force the example CUDA wheel if a newer compatible PyTorch installation already works.

The original Fedora workstation used a CUDA 13.1 system toolkit. That is a supported
project target. A PyTorch wheel's suffix describes the CUDA runtime bundled with that
wheel; it does not need to be textually identical to the system toolkit version. What
matters is that the NVIDIA driver is new enough and every compiled extension passes a
real forward/backward capability test.

For an existing CUDA 13.1 Fedora environment, preserve the working PyTorch build and
compile Transformer Engine explicitly against that toolkit:

```bash
export CUDA_HOME=/usr/local/cuda-13.1
ASTERLM_VENV_PATH="$VIRTUAL_ENV" \
  bash scripts/install_transformer_engine_linux.sh
PYTHONPATH=src python scripts/frontier_vnext_capabilities.py \
  --output runs/setup/capabilities-fedora-cuda-13.1.json
```

In the recovered WSL environment, PyTorch currently reports CUDA runtime 13.0 while
the NVIDIA compiler wheel is 13.3. This is a separate validation environment, not a
claim that the Fedora machine used those versions. Aster detects the wheel toolkit's
headers and configures Transformer Engine's NVRTC include path without overwriting an
explicit Fedora `CUDA_HOME`/`NVTE_CUDA_INCLUDE_DIR`.

## Verify

```bash
python scripts/hardware_probe.py --output runs/hardware-probe.json
python scripts/system_check.py --model configs/model/aster_moe_frontier_893m_a484m.yaml
pytest
python scripts/smoke_train.py
```

## Ada precision reality

The RTX 4080 Laptop GPU is Ada, compute capability 8.9. NVIDIA Transformer Engine supports FP8 on Ada. NVFP4 is a Blackwell feature. AsterLM therefore tests FP8 execution and uses low-bit storage/state methods for four-bit VRAM savings rather than claiming native FP4 tensor-core training.

## Laptop reproducibility

For every benchmark:

- use the original power adapter
- use one fixed performance profile
- disable sleep
- record GPU temperature, power, core/memory clocks
- allow compile/warmup before timing
- report sustained rather than first-minute throughput
- avoid browser/video/other CUDA workloads

## Memory allocator

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
```

`expandable_segments` can reduce fragmentation; it cannot make an oversized computation fit.

## FLA and Transformer Engine

FLA/KDA and TE are optional compiled research dependencies. ABI/version mismatch is common. An import alone is insufficient: run `scripts/frontier_vnext_capabilities.py`, which executes FP8 delayed/current scaling, FLA CUDA, and grouped-MoE forward/backward probes.

If Transformer Engine fails, use the shape-identical Torch model and BF16. If FLA fails, the PyTorch KDA fallback can debug correctness but is not a viable serious-training speed path.

## Remote provider clients

Install the optional remote-control clients into the isolated environment rather than the system
Python. Modal is pinned to the currently validated 1.5 client series:

```bash
python -m pip install -e '.[remote]'
modal profile list --json
```

Modal profiles represent authorized Workspaces. Select one per process with `MODAL_PROFILE`; never
copy token values into Studio settings, run contracts, logs, or Git. Workspace membership/invites are
the supported collaboration path when friends contribute compute.

On Windows, `ASTER_STUDIO.ps1` maps the existing `%USERPROFILE%\.modal.toml` into WSL through
`MODAL_CONFIG_PATH`. This keeps one provider-native credential store, makes every authorized profile
alias visible in Studio, and avoids copying secrets into the repository or WSL home directory.

## OOM escalation

1. micro-batch 1
2. checkpointing
3. 4/8-bit optimizer states
4. activation offload
5. CPU optimizer offload
6. LoQT INT4 FFN/expert storage
7. shorter training block length
8. smaller architecture

The order deliberately spends additional compute/PCIe bandwidth before sacrificing capacity.

## Thermal failure

If tokens/s declines over several minutes while memory remains stable, inspect power and temperature logs. Lower sustained clocks may make a larger model impractically slow even when it fits. That is a wall-clock decision, not a correctness failure.
