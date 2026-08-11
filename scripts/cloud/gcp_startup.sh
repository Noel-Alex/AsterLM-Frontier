#!/usr/bin/env bash
set -euo pipefail

metadata() {
  curl -fsS -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"
}

CONTRACT_URI="$(metadata aster-contract-uri)"
IMAGE="$(metadata aster-artifact-image)"
BUCKET="$(metadata aster-bucket)"
HF_SECRET="$(metadata aster-hf-secret)"
WANDB_SECRET="$(metadata aster-wandb-secret)"
install -d -m 0700 /var/lib/aster-run /var/lib/aster-cache /mnt/aster-gcs
gcloud storage cp "$CONTRACT_URI" /var/lib/aster-run/contract.json

if command -v gcsfuse >/dev/null 2>&1; then
  gcsfuse --implicit-dirs --cache-dir=/var/lib/aster-cache "$BUCKET" /mnt/aster-gcs
fi

HF_TOKEN="$(gcloud secrets versions access latest --secret="$HF_SECRET")"
WANDB_API_KEY="$(gcloud secrets versions access latest --secret="$WANDB_SECRET")"
export HF_TOKEN WANDB_API_KEY ASTERLM_REMOTE_PROVIDER=gcp
export ASTERLM_REMOTE_CONTRACT=/run/aster/contract.json

docker pull "$IMAGE"
docker run --rm --gpus all --name aster-train --stop-timeout 100 \
  -e HF_TOKEN -e WANDB_API_KEY -e ASTERLM_REMOTE_PROVIDER -e ASTERLM_REMOTE_CONTRACT \
  -v /var/lib/aster-run:/run/aster \
  -v /var/lib/aster-cache:/var/cache/aster \
  -v /mnt/aster-gcs:/mnt/aster-gcs \
  "$IMAGE" python scripts/cloud/run_contract.py --contract /run/aster/contract.json
