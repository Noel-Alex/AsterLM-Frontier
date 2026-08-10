#!/usr/bin/env bash
set -euo pipefail

# Reproducible Transformer Engine PyTorch binding install for Fedora/WSL.
# A system CUDA toolkit (for example Fedora's CUDA 13.1) wins when CUDA_HOME is
# supplied. Otherwise, the NVIDIA CUDA wheel toolkit inside the active venv is
# discovered without mutating /usr/local/cuda.

ASTERLM_VENV_PATH="${ASTERLM_VENV_PATH:-/root/.venvs/asterlm}"
TE_VERSION="${ASTERLM_TE_VERSION:-2.17.0}"
TE_BUILD_JOBS="${ASTERLM_TE_BUILD_JOBS:-8}"

if [[ ! -f "${ASTERLM_VENV_PATH}/bin/activate" ]]; then
  echo "AsterLM venv not found: ${ASTERLM_VENV_PATH}" >&2
  exit 2
fi

# shellcheck disable=SC1090
source "${ASTERLM_VENV_PATH}/bin/activate"

if [[ -n "${ASTERLM_UV_PATH:-}" && -x "${ASTERLM_UV_PATH}" ]]; then
  installer=("${ASTERLM_UV_PATH}" pip)
elif command -v uv >/dev/null 2>&1; then
  installer=("$(command -v uv)" pip)
elif [[ -x /root/.local/bin/uv ]]; then
  installer=(/root/.local/bin/uv pip)
elif python -m pip --version >/dev/null 2>&1; then
  installer=(python -m pip)
else
  echo "Neither uv nor pip is available in ${ASTERLM_VENV_PATH}." >&2
  exit 2
fi

"${installer[@]}" install ninja nvidia-cuda-cccl
"${installer[@]}" install \
  "transformer-engine==${TE_VERSION}" \
  "transformer-engine-cu13==${TE_VERSION}"

if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
  cuda_root="${CUDA_HOME}"
elif command -v nvcc >/dev/null 2>&1; then
  cuda_root="$(dirname "$(dirname "$(command -v nvcc)")")"
else
  cuda_root="$(python -c 'import site; from pathlib import Path; roots=[Path(p)/"nvidia/cu13" for p in site.getsitepackages()]; print(next(str(p) for p in roots if (p/"bin/nvcc").is_file()))')"
fi

nvidia_root="$(python -c 'import site; from pathlib import Path; roots=[Path(p)/"nvidia" for p in site.getsitepackages()]; print(next(str(p) for p in roots if p.is_dir()))')"
mapfile -t include_paths < <(
  find "${nvidia_root}" -mindepth 2 -maxdepth 2 -type d -name include -print | sort
)
mapfile -t library_paths < <(
  find "${nvidia_root}" -mindepth 2 -maxdepth 2 -type d -name lib -print | sort
)
if (( ${#include_paths[@]} == 0 || ${#library_paths[@]} == 0 )); then
  echo "Could not discover NVIDIA wheel include/library directories under ${nvidia_root}" >&2
  exit 2
fi

export CUDA_HOME="${cuda_root}"
export CPATH="$(IFS=:; echo "${include_paths[*]}")${CPATH:+:${CPATH}}"
export LIBRARY_PATH="$(IFS=:; echo "${library_paths[*]}")${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export LD_LIBRARY_PATH="${cuda_root}/lib64:$(IFS=:; echo "${library_paths[*]}")${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export MAX_JOBS="${TE_BUILD_JOBS}"
export NVTE_FRAMEWORK=pytorch

echo "Building transformer-engine-torch ${TE_VERSION}"
echo "Python: $(python --version 2>&1)"
echo "CUDA_HOME: ${CUDA_HOME}"
"${CUDA_HOME}/bin/nvcc" --version | tail -n 1

"${installer[@]}" install --no-build-isolation "transformer-engine-torch==${TE_VERSION}"

python -c 'import torch; import transformer_engine.pytorch as te; print("Transformer Engine PyTorch binding:", te.__file__); print("CUDA:", torch.version.cuda, torch.cuda.get_device_name(0))'
