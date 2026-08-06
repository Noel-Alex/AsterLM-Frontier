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

FLA/KDA and TE are optional compiled research dependencies. ABI/version mismatch is common. The setup script tests imports, but the definitive test is a forward/backward profiler run.

If Transformer Engine fails, use the shape-identical Torch model and BF16. If FLA fails, the PyTorch KDA fallback can debug correctness but is not a viable serious-training speed path.

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
