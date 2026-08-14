#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 <watched-pid> <output-directory>" >&2
}

if [[ $# -ne 2 || ! $1 =~ ^[1-9][0-9]*$ ]]; then
  usage
  exit 2
fi

watched_pid=$1
output_dir=$2

if [[ ! -r "/proc/${watched_pid}/stat" ]]; then
  echo "watched process ${watched_pid} is not running" >&2
  exit 1
fi

mkdir -p "${output_dir}"
watched_start_ticks=$(awk '{print $22}' "/proc/${watched_pid}/stat")
dmon_log="${output_dir}/continuous-gpu-dmon.log"
dmon_pid_file="${output_dir}/continuous-gpu-dmon.pid"
watcher_pid_file="${output_dir}/continuous-gpu-watcher.pid"

printf '%s\n' "$$" > "${watcher_pid_file}"
date --iso-8601=seconds > "${output_dir}/continuous-gpu-dmon-start.txt"

nvidia-smi dmon -s pucm -d 1 >> "${dmon_log}" 2>&1 &
dmon_pid=$!
printf '%s\n' "${dmon_pid}" > "${dmon_pid_file}"

cleanup() {
  kill "${dmon_pid}" 2>/dev/null || true
  wait "${dmon_pid}" 2>/dev/null || true
  rm -f "${dmon_pid_file}" "${watcher_pid_file}"
}
trap cleanup EXIT INT TERM

while [[ -r "/proc/${watched_pid}/stat" ]]; do
  current_start_ticks=$(awk '{print $22}' "/proc/${watched_pid}/stat")
  [[ ${current_start_ticks} == "${watched_start_ticks}" ]] || break
  kill -0 "${dmon_pid}" 2>/dev/null || {
    echo "nvidia-smi dmon exited before the watched process" >&2
    exit 1
  }
  sleep 5
done
