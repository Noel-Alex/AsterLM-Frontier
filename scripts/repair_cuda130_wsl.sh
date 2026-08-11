#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${ASTER_PYTHON:-/root/.venvs/asterlm/bin/python}"
UV_BIN="${ASTER_UV:-/root/.local/bin/uv}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"${UV_BIN}" pip install \
  --python "${PYTHON_BIN}" \
  --constraint "${ROOT}/constraints/cuda130-wsl.txt" \
  -r "${ROOT}/constraints/cuda130-wsl.txt"

"${PYTHON_BIN}" "${ROOT}/scripts/check_cuda_toolchain.py" --require-compatible
