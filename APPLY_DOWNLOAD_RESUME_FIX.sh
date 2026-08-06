#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

if [[ ! -f pyproject.toml || ! -f scripts/materialize_corpus.py ]]; then
  echo "Run this script from the patched AsterLM-Frontier checkout." >&2
  exit 2
fi

python - <<'PY'
from pathlib import Path

path = Path('.gitignore')
text = path.read_text(encoding='utf-8')
lines = text.splitlines()
changed = False
for i, line in enumerate(lines):
    if line == 'data/':
        lines[i] = '/data/'
        changed = True
if changed:
    suffix = '\n' if text.endswith('\n') else ''
    path.write_text('\n'.join(lines) + suffix, encoding='utf-8')
    print('Fixed .gitignore: only the root runtime /data/ directory is ignored.')
elif '/data/' in lines:
    print('.gitignore root /data/ rule is already correct.')
else:
    raise SystemExit('Could not find the expected data/ ignore rule; inspect .gitignore manually.')
PY

export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest -q \
  tests/test_hf_stream.py \
  tests/test_resumable_downloads.py \
  tests/test_100b_campaign.py

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if git check-ignore -q src/asterlm/data/__init__.py; then
    echo "ERROR: src/asterlm/data is still ignored by Git." >&2
    exit 3
  fi
  if git check-ignore -q configs/data/pretrain_frontier_local_18p4b.yaml; then
    echo "ERROR: configs/data is still ignored by Git." >&2
    exit 3
  fi
else
  echo "Warning: no .git directory in this copy; Git ignore checks were skipped."
fi

cat <<'EOF'

Patch validation passed.

Inspect the existing FineWeb cursor:
  python scripts/inspect_download_cursor.py data/corpus-frontier-16b/fineweb_edu

Restart with disk + RAM protection (command-level relaunch disabled):
  ./RUN_100B_SAFE.sh

Before your next push, make sure Git tracks the previously ignored package/config directories:
  git add .gitignore src/asterlm/data configs/data scripts tests/test_hf_stream.py docs/DOWNLOAD_RESUME_MEMORY_FIX.md APPLY_DOWNLOAD_RESUME_FIX.sh
EOF
