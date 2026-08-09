#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "AsterLM Studio expected the project environment at:" >&2
  echo "  $ROOT/.venv/bin/python" >&2
  echo "Create/repair the AsterLM environment first." >&2
  exit 2
fi

HOST="${ASTER_STUDIO_HOST:-127.0.0.1}"
PORT="${ASTER_STUDIO_PORT:-8765}"

exec .venv/bin/python studio/server.py --host "$HOST" --port "$PORT" "$@"
