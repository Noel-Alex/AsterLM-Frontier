#!/usr/bin/env bash
set -u

if docker inspect aster-train >/dev/null 2>&1; then
  docker stop --time 100 aster-train || true
fi
if mountpoint -q /mnt/aster-gcs; then
  fusermount -u /mnt/aster-gcs || true
fi
