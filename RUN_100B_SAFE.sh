#!/usr/bin/env bash
set -uo pipefail

# Override at launch, e.g. MIN_FREE_GIB=180 MAX_RSS_GIB=18 ./RUN_100B_SAFE.sh
MIN_FREE_GIB="${MIN_FREE_GIB:-150}"
MAX_RSS_GIB="${MAX_RSS_GIB:-20}"
POLL_SECONDS="${POLL_SECONDS:-30}"

python scripts/download_data.py \
  --profile overtrain100 \
  --require-auth \
  --network-mode low \
  --max-retries 0 \
  --command-retries 0 \
  --max-rss-gib "$MAX_RSS_GIB" &

download_pid=$!
stopping=0

graceful_stop() {
    if (( stopping )); then
        return
    fi
    stopping=1
    echo
    echo "Stopping AsterLM downloader gracefully..."
    kill -INT "$download_pid" 2>/dev/null || true
}

trap graceful_stop INT TERM

while kill -0 "$download_pid" 2>/dev/null; do
    available_bytes=$(df -PB1 . | awk 'NR == 2 {print $4}')
    available_gib=$((available_bytes / 1024 / 1024 / 1024))

    printf '\rFree disk: %4d GiB | floor: %d GiB | materializer RSS ceiling: %s GiB' \
        "$available_gib" "$MIN_FREE_GIB" "$MAX_RSS_GIB"

    if (( available_gib < MIN_FREE_GIB )); then
        echo
        echo "Disk safety floor reached; requesting a cursor + shard checkpoint."
        graceful_stop
        break
    fi

    sleep "$POLL_SECONDS"
done

wait "$download_pid"
status=$?
echo

# 75 is an intentional memory/disk safety stop. Rerunning this script resumes.
if (( status == 75 )); then
    echo "AsterLM stopped at a safety boundary. The next run will resume the saved cursor."
fi
exit "$status"
