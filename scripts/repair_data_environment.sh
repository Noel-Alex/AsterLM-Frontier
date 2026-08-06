#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  if [[ ! -f .venv/bin/activate ]]; then
    echo "Run this from the AsterLM-Frontier repository root." >&2
    exit 2
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

python -m pip install --upgrade \
  "huggingface_hub>=0.32.0" \
  "datasets>=3.5" \
  "fsspec>=2024.9" \
  "zstandard>=0.23" \
  "pyarrow>=18"

# Refresh the editable package and base data dependencies without replacing the
# user's known-good CUDA PyTorch build or recompiling optional CUDA packages.
python -m pip install -e .

python scripts/data_preflight.py --minimum-free-gib "${ASTER_MIN_FREE_GIB:-90}"

if command -v hf >/dev/null 2>&1; then
  if ! hf auth whoami >/dev/null 2>&1; then
    echo
    echo "Hugging Face authentication is not configured. Recommended next command:"
    echo "  hf auth login"
  fi
else
  echo "Warning: the 'hf' CLI was not found after installation." >&2
fi

echo
echo "Data environment repaired. Recommended command:"
echo "  python scripts/download_data.py --profile all --validate-first --network-mode low"
