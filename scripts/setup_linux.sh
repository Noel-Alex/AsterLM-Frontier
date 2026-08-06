#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
WITH_FP8=0
WITH_TORCHAO=0
WITH_TRACKING=0
WITH_APOLLO=0
WITH_REASONING=0
SKIP_TORCH=0

usage() {
  cat <<'EOF'
Usage: bash scripts/setup_linux.sh [options]

Options:
  --with-apollo     Install APOLLO/APOLLO-Mini optimizer support.
  --with-fp8        Install NVIDIA Transformer Engine for FP8 experiments.
  --with-torchao    Install TorchAO quantization/low-bit optimizer support.
  --with-tracking   Install Weights & Biases and TensorBoard.
  --with-reasoning  Install math verifier and symbolic-equivalence support.
  --skip-torch      Keep the PyTorch already installed in .venv.
  -h, --help        Show this help.

Environment variables:
  PYTHON_BIN        Python executable used to create .venv (default: python3.12).
  TORCH_INDEX_URL   PyTorch wheel index (default: CUDA 12.6 index).
EOF
}

for arg in "$@"; do
  case "$arg" in
    --with-apollo) WITH_APOLLO=1 ;;
    --with-fp8) WITH_FP8=1 ;;
    --with-torchao) WITH_TORCHAO=1 ;;
    --with-tracking) WITH_TRACKING=1 ;;
    --with-reasoning) WITH_REASONING=1 ;;
    --skip-torch) SKIP_TORCH=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! -d .venv ]]; then
  "$PYTHON_BIN" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

if [[ "$SKIP_TORCH" == "0" ]]; then
  python -m pip install --upgrade torch --index-url "$TORCH_INDEX_URL"
else
  python - <<'PY_CHECK'
import torch
print(f"Keeping existing PyTorch {torch.__version__}")
PY_CHECK
fi

# Core data, training, CUDA/KDA, and test dependencies. Optional VRAM paths
# are installed independently so a failed research package does not poison
# the base environment.
python -m pip install -e ".[cuda,dev]"

if [[ "$WITH_APOLLO" == "1" ]]; then
  python -m pip install -e ".[memory]"
fi
if [[ "$WITH_TORCHAO" == "1" ]]; then
  python -m pip install -e ".[quant]"
fi
if [[ "$WITH_FP8" == "1" ]]; then
  python -m pip install -e ".[fp8]"
fi
if [[ "$WITH_TRACKING" == "1" ]]; then
  python -m pip install -e ".[tracking]"
fi
if [[ "$WITH_REASONING" == "1" ]]; then
  python -m pip install -e ".[reasoning]"
  if ! command -v bwrap >/dev/null 2>&1; then
    echo "WARNING: bubblewrap is absent; Python RLVR rewards will refuse to execute untrusted code." >&2
    echo "Fedora: sudo dnf install bubblewrap" >&2
  fi
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

python - <<'PY_DATA_CHECK'
import zstandard  # registers the codec with fsspec
from fsspec.compression import compr
from datasets import Dataset

if "zstd" not in compr:
    raise SystemExit("zstd was not registered with fsspec")
stream = Dataset.from_dict({"x": [1, 2]}).to_iterable_dataset(num_shards=2)
iterator = iter(stream)
next(iterator)
state = stream.state_dict()
restored = Dataset.from_dict({"x": [1, 2]}).to_iterable_dataset(num_shards=2)
restored.load_state_dict(state)
print("Data codecs and shard-aware resume support: OK")
PY_DATA_CHECK

python scripts/system_check.py --model configs/model/aster_moe_frontier_893m_a484m.yaml
python -m pytest

echo "AsterLM Frontier is ready. Activate with: source .venv/bin/activate"
echo "Data guide: docs/LOW_BANDWIDTH_DOWNLOADS.md"
echo "Next: python scripts/hardware_probe.py && python scripts/run_frontier_experiments.py --mode quick"
