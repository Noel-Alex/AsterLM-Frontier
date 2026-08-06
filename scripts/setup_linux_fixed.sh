#!/usr/bin/env bash
set -Eeuo pipefail

# AsterLM Frontier Linux setup
#
# Key behavior:
#   * Does not hard-code a Python executable that may not exist.
#   * Reuses a healthy .venv, repairs one that only lacks pip, and removes a
#     partially-created/broken .venv.
#   * Creates the environment with --without-pip first, avoiding Fedora builds
#     where `venv` succeeds but the bundled ensurepip step fails.
#   * Bootstraps pip with ensurepip when possible, otherwise with get-pip.py.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
PYTHON_BIN="${PYTHON_BIN:-}"
MIN_PYTHON_MAJOR=3
MIN_PYTHON_MINOR=10

WITH_FP8=0
WITH_TORCHAO=0
WITH_TRACKING=0
WITH_APOLLO=0
SKIP_TORCH=0
RECREATE_VENV=0
SKIP_CHECKS=0

usage() {
  cat <<'USAGE'
Usage: bash scripts/setup_linux.sh [options]

Options:
  --with-apollo       Install APOLLO/APOLLO-Mini optimizer support.
  --with-fp8          Install NVIDIA Transformer Engine for FP8 experiments.
  --with-torchao      Install TorchAO quantization/low-bit optimizer support.
  --with-tracking     Install Weights & Biases and TensorBoard.
  --skip-torch        Keep the PyTorch already installed in .venv.
  --recreate-venv     Delete and recreate .venv.
  --skip-checks       Skip system_check.py and pytest after installation.
  -h, --help          Show this help.

Environment variables:
  PYTHON_BIN          Preferred Python executable, for example python3.13.
                      If unavailable or unset, a compatible installed Python
                      is selected automatically.
  TORCH_INDEX_URL     PyTorch wheel index
                      (default: https://download.pytorch.org/whl/cu126).
USAGE
}

for arg in "$@"; do
  case "$arg" in
    --with-apollo) WITH_APOLLO=1 ;;
    --with-fp8) WITH_FP8=1 ;;
    --with-torchao) WITH_TORCHAO=1 ;;
    --with-tracking) WITH_TRACKING=1 ;;
    --skip-torch) SKIP_TORCH=1 ;;
    --recreate-venv) RECREATE_VENV=1 ;;
    --skip-checks) SKIP_CHECKS=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; usage >&2; exit 2 ;;
  esac
done

on_error() {
  local exit_code=$?
  local line_no=${1:-unknown}
  echo >&2
  echo "Setup failed at line ${line_no} (exit code ${exit_code})." >&2
  echo "The existing .venv was left in place for inspection." >&2
  exit "$exit_code"
}
trap 'on_error $LINENO' ERR

python_is_compatible() {
  local executable=$1
  "$executable" - <<PY_CHECK >/dev/null 2>&1
import sys
minimum = (${MIN_PYTHON_MAJOR}, ${MIN_PYTHON_MINOR})
raise SystemExit(0 if sys.version_info[:2] >= minimum else 1)
PY_CHECK
}

resolve_python() {
  local -a candidates=()
  local candidate resolved

  if [[ -n "$PYTHON_BIN" ]]; then
    candidates+=("$PYTHON_BIN")
  fi

  # Prefer versions with broad ML-package compatibility, then fall back to the
  # Fedora system interpreter. Duplicates are harmless and skipped below.
  candidates+=(python3.12 python3.13 python3.11 python3.14 python3 python)

  local -A seen=()
  for candidate in "${candidates[@]}"; do
    [[ -n "$candidate" ]] || continue
    if [[ -n "${seen[$candidate]:-}" ]]; then
      continue
    fi
    seen[$candidate]=1

    if ! resolved="$(command -v -- "$candidate" 2>/dev/null)"; then
      if [[ -n "$PYTHON_BIN" && "$candidate" == "$PYTHON_BIN" ]]; then
        echo "Warning: requested PYTHON_BIN='$PYTHON_BIN' was not found; trying installed alternatives." >&2
      fi
      continue
    fi

    if python_is_compatible "$resolved"; then
      printf '%s\n' "$resolved"
      return 0
    fi
  done

  echo "Error: no compatible Python >= ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR} was found." >&2
  echo "On Fedora, install one with: sudo dnf install python3.13" >&2
  return 1
}

bootstrap_pip() {
  local venv_python=$1
  local bootstrap_file

  if "$venv_python" -m pip --version >/dev/null 2>&1; then
    return 0
  fi

  echo "pip is missing; attempting ensurepip..."
  if "$venv_python" -m ensurepip --upgrade --default-pip >/dev/null 2>&1; then
    return 0
  fi

  echo "ensurepip is unavailable or broken; bootstrapping pip with get-pip.py..."
  bootstrap_file="$(mktemp --suffix=-get-pip.py)"

  if command -v curl >/dev/null 2>&1; then
    curl --fail --location --silent --show-error \
      https://bootstrap.pypa.io/get-pip.py \
      --output "$bootstrap_file"
  elif command -v wget >/dev/null 2>&1; then
    wget --quiet https://bootstrap.pypa.io/get-pip.py -O "$bootstrap_file"
  else
    rm -f "$bootstrap_file"
    echo "Error: pip is unavailable and neither curl nor wget is installed." >&2
    echo "Install curl with: sudo dnf install curl" >&2
    return 1
  fi

  "$venv_python" "$bootstrap_file"
  rm -f "$bootstrap_file"
  "$venv_python" -m pip --version >/dev/null
}

if [[ "$RECREATE_VENV" == "1" && -e .venv ]]; then
  echo "Removing existing .venv..."
  rm -rf .venv
fi

VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"

if [[ -e .venv && ! -x "$VENV_PYTHON" ]]; then
  echo "Removing incomplete .venv (no working Python executable)..."
  rm -rf .venv
fi

if [[ ! -d .venv ]]; then
  SELECTED_PYTHON="$(resolve_python)"
  echo "Creating .venv with: $SELECTED_PYTHON ($("$SELECTED_PYTHON" --version 2>&1))"

  # Avoid venv's automatic ensurepip phase. Some Fedora Python installations
  # fail there even though the interpreter and venv module themselves work.
  "$SELECTED_PYTHON" -m venv --without-pip .venv
fi

if ! "$VENV_PYTHON" -c 'import sys; print(sys.executable)' >/dev/null 2>&1; then
  echo "Existing .venv is broken; recreating it..."
  rm -rf .venv
  SELECTED_PYTHON="$(resolve_python)"
  "$SELECTED_PYTHON" -m venv --without-pip .venv
fi

if ! python_is_compatible "$VENV_PYTHON"; then
  echo "Error: .venv uses an unsupported Python version." >&2
  echo "Run again with --recreate-venv." >&2
  exit 1
fi

bootstrap_pip "$VENV_PYTHON"

PYTHON="$VENV_PYTHON"
PIP=("$PYTHON" -m pip)

echo "Using $($PYTHON --version 2>&1) at $PYTHON"
"${PIP[@]}" install --upgrade pip setuptools wheel

if [[ "$SKIP_TORCH" == "0" ]]; then
  "${PIP[@]}" install --upgrade torch --index-url "$TORCH_INDEX_URL"
else
  "$PYTHON" - <<'PY_CHECK'
import torch
print(f"Keeping existing PyTorch {torch.__version__}")
PY_CHECK
fi

# Core data, training, CUDA/KDA, and test dependencies.
"${PIP[@]}" install -e ".[cuda,dev]"

if [[ "$WITH_APOLLO" == "1" ]]; then
  "${PIP[@]}" install -e ".[memory]"
fi
if [[ "$WITH_TORCHAO" == "1" ]]; then
  "${PIP[@]}" install -e ".[quant]"
fi
if [[ "$WITH_FP8" == "1" ]]; then
  "${PIP[@]}" install -e ".[fp8]"
fi
if [[ "$WITH_TRACKING" == "1" ]]; then
  "${PIP[@]}" install -e ".[tracking]"
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [[ "$SKIP_CHECKS" == "0" ]]; then
  "$PYTHON" scripts/system_check.py \
    --model configs/model/aster_moe_frontier_893m_a484m.yaml
  "$PYTHON" -m pytest
fi

echo
echo "AsterLM Frontier is ready."
echo "Activate with: source .venv/bin/activate"
echo "Next: python scripts/hardware_probe.py && python scripts/run_frontier_experiments.py --mode quick"
