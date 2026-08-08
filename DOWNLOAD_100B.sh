#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
if [[ ! -x ".venv/bin/python" ]]; then
  echo "AsterLM .venv was not found at $ROOT/.venv" >&2
  exit 2
fi
exec .venv/bin/python scripts/download_100b.py "$@"
