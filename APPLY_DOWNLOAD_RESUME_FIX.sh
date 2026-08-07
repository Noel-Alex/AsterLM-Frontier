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
python -m compileall -q src scripts tests
python -m pytest -q tests/test_hf_stream.py

cat <<'EOF2'

v7 validation passed.

Important: --dry-run never changes the cursor. If FineWeb is currently stopped,
preview and then commit the latest next-shard conversion:
  python scripts/migrate_legacy_cursor.py data/corpus-frontier-16b/fineweb_edu --dry-run
  python scripts/migrate_legacy_cursor.py data/corpus-frontier-16b/fineweb_edu

Inspect it:
  python scripts/inspect_download_cursor.py data/corpus-frontier-16b/fineweb_edu

Recommended normal run (new default: balanced):
  ./RUN_100B_SAFE.sh

Faster while preserving one active decoded dataset shard:
  ./RUN_100B_SAFE.sh --network-mode safe-fast

Avoid --network-mode fast on a 32 GiB machine unless you intentionally want
Hugging Face high-performance mode and have verified memory headroom.
EOF2
