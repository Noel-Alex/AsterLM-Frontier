#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 COMMIT OUTPUT_ROOT MODEL_CONFIG TRAIN_CONFIG" >&2
  exit 2
fi

commit="$1"
output_root="$2"
model_config="$3"
train_config="$4"
segments="${ASTER_SWEEP_SEGMENTS:-2,4,8}"
segments="${segments//,/ }"
steps="${ASTER_SWEEP_STEPS:-20}"
warmup="${ASTER_SWEEP_WARMUP:-5}"
repetitions="${ASTER_SWEEP_REPETITIONS:-2}"
sequence="${ASTER_SWEEP_SEQUENCE:-2048}"
batch="${ASTER_SWEEP_BATCH:-4}"
accum="${ASTER_SWEEP_ACCUM:-4}"
optimizer="${ASTER_SWEEP_OPTIMIZER:-adamw}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
short_commit="$(git -C "$repo_root" rev-parse --short=12 "$commit")"
worktree="/tmp/aster-bench-${short_commit}"

if [[ ! -e "$worktree/.git" ]]; then
  git -C "$repo_root" worktree prune
  git -C "$repo_root" worktree add --detach "$worktree" "$commit"
fi

actual_commit="$(git -C "$worktree" rev-parse HEAD)"
expected_commit="$(git -C "$repo_root" rev-parse "$commit^{commit}")"
if [[ "$actual_commit" != "$expected_commit" ]]; then
  echo "worktree commit mismatch: expected=$expected_commit actual=$actual_commit" >&2
  exit 3
fi
if [[ -n "$(git -C "$worktree" status --porcelain --untracked-files=all)" ]]; then
  echo "benchmark worktree is dirty: $worktree" >&2
  git -C "$worktree" status --short >&2
  exit 4
fi

export PYTHONPATH="$worktree/src"
cd "$worktree"
python_bin="/root/.venvs/asterlm/bin/python"
imported_root="$($python_bin -c 'from asterlm.source_provenance import imported_repo_root; print(imported_repo_root())')"
if [[ "$imported_root" != "$worktree" ]]; then
  echo "AsterLM import mismatch: expected=$worktree imported=$imported_root" >&2
  exit 5
fi
if [[ "${ASTER_SWEEP_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  printf 'source-pinned: commit=%s worktree=%s imported=%s\n' "$actual_commit" "$worktree" "$imported_root"
  exit 0
fi

model_root="$(dirname "$model_config")"
model_filename="$(basename "$model_config")"
mkdir -p "$output_root"

for segment in $segments; do
  if [[ ! "$segment" =~ ^[1-9][0-9]*$ ]]; then
    echo "invalid checkpoint segment size: $segment" >&2
    exit 7
  fi
  target="$output_root/segment${segment}"
  if [[ -e "$target" ]]; then
    echo "refusing to overwrite existing benchmark output: $target" >&2
    exit 6
  fi
  "$python_bin" scripts/run_moe_utilization_matrix.py \
    --config-root "$model_root" \
    --train-config "$train_config" \
    --output "$target" \
    --steps "$steps" \
    --warmup "$warmup" \
    --repetitions "$repetitions" \
    --sequence "$sequence" \
    --batch "$batch" \
    --accum "$accum" \
    --optimizer "$optimizer" \
    --checkpoint-segment-size "$segment" \
    --variant-spec "k3-cutlass=${model_filename}=cutlass" \
    --variants k3-cutlass
done
