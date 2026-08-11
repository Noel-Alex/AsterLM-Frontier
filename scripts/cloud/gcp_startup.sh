#!/usr/bin/env bash
set -euo pipefail

metadata() {
  curl -fsS -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"
}

optional_metadata() {
  curl -fsS -H 'Metadata-Flavor: Google' \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1" 2>/dev/null || true
}

CONTRACT_URI="$(metadata aster-contract-uri)"
IMAGE="$(metadata aster-artifact-image)"
BUCKET="$(metadata aster-bucket)"
HF_SECRET="$(metadata aster-hf-secret)"
WANDB_SECRET="$(metadata aster-wandb-secret)"
CONTRACT_ID="$(metadata aster-contract-id)"
INSTANCE_NAME="$(metadata aster-instance-name)"
INSTANCE_ZONE="$(metadata aster-instance-zone)"
PROJECT_ID="$(metadata aster-project-id)"
install -d -m 0700 /var/lib/aster-run /var/cache/gcsfuse /mnt/aster-gcs

finish() {
  code=$?
  trap - EXIT
  if [[ -d /var/lib/aster-run ]]; then
    printf '{"contract_id":"%s","exit_code":%d,"finished_utc":"%s"}\n' \
      "$CONTRACT_ID" "$code" "$(date -u +%FT%TZ)" >/var/lib/aster-run/exit.json
    gcloud storage cp --recursive /var/lib/aster-run \
      "gs://$BUCKET/logs/$CONTRACT_ID/" \
      --project="$PROJECT_ID" || true
  fi
  # The VM service account receives a condition-scoped permission to delete only
  # this named instance. The hard max-run-duration DELETE policy remains the
  # independent billing backstop if this early teardown cannot complete.
  gcloud compute instances delete "$INSTANCE_NAME" \
    --zone="$INSTANCE_ZONE" --project="$PROJECT_ID" \
    --delete-disks=all --quiet || shutdown -h now
  exit "$code"
}
trap finish EXIT

for required in curl docker gcloud gcsfuse nvidia-smi mountpoint; do
  command -v "$required" >/dev/null 2>&1 || {
    echo "Qualified host image is missing required command: $required" >&2
    exit 1
  }
done
nvidia-smi -L >/dev/null
docker info >/dev/null

gcloud storage cp "$CONTRACT_URI" /var/lib/aster-run/contract.json

gcsfuse --implicit-dirs --file-cache-enable-parallel-downloads \
  --file-cache-max-size-mb=-1 --cache-dir=/var/cache/gcsfuse \
  "$BUCKET" /mnt/aster-gcs
test -d "/mnt/aster-gcs/datasets"
install -d -m 0700 "/mnt/aster-gcs/cache" "/mnt/aster-gcs/checkpoints"

HF_TOKEN="$(gcloud secrets versions access latest --secret="$HF_SECRET")"
WANDB_API_KEY="$(gcloud secrets versions access latest --secret="$WANDB_SECRET")"
export HF_TOKEN WANDB_API_KEY ASTERLM_REMOTE_PROVIDER=gcp
export ASTERLM_REMOTE_CONTRACT=/run/aster/contract.json
export ASTERLM_REMOTE_RUN_ROOT=/run/aster/runs
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/var/cache/aster/huggingface
export TRITON_CACHE_DIR=/var/cache/aster/triton
export TORCH_EXTENSIONS_DIR=/var/cache/aster/torch_extensions
export XDG_CACHE_HOME=/var/cache/aster/xdg

docker pull "$IMAGE"
docker run --rm --gpus all --name aster-train --stop-timeout 100 \
  -e HF_TOKEN -e WANDB_API_KEY -e ASTERLM_REMOTE_PROVIDER -e ASTERLM_REMOTE_CONTRACT \
  -e ASTERLM_REMOTE_RUN_ROOT -e PYTORCH_ALLOC_CONF -e PYTORCH_CUDA_ALLOC_CONF \
  -e HF_HOME -e TRITON_CACHE_DIR -e TORCH_EXTENSIONS_DIR -e XDG_CACHE_HOME \
  -v /var/lib/aster-run:/run/aster \
  -v /mnt/aster-gcs/datasets:/opt/aster/data:ro \
  -v /mnt/aster-gcs/cache:/var/cache/aster \
  "$IMAGE" python scripts/cloud/run_contract.py --contract /run/aster/contract.json &
CONTAINER_WAIT_PID=$!

watch_stop_request() {
  while kill -0 "$CONTAINER_WAIT_PID" 2>/dev/null; do
    if [[ "$(optional_metadata aster-stop-request)" == "graceful" ]]; then
      # run_contract.py forwards SIGTERM to the trainer process group. Studio then
      # stops only at an optimizer boundary and verifies the durable checkpoint.
      docker kill --signal=TERM aster-train >/dev/null
      return
    fi
    sleep 5
  done
}
watch_stop_request &
WATCH_PID=$!
set +e
wait "$CONTAINER_WAIT_PID"
CONTAINER_CODE=$?
set -e
kill "$WATCH_PID" 2>/dev/null || true
wait "$WATCH_PID" 2>/dev/null || true
exit "$CONTAINER_CODE"
