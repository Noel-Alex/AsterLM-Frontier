#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

# Keep the foreground supervisor attached to the terminal. It creates a separate
# session for download_data.py and every materializer, then forwards signals to
# that entire process group with bounded SIGINT -> SIGTERM -> SIGKILL escalation.
exec "${PYTHON:-python}" scripts/run_100b_safe.py "$@"
