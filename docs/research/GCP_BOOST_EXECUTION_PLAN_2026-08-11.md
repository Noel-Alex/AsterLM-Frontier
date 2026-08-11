# Google Cloud boosted training execution plan

Date: 2026-08-11

## Decision

Google Cloud is a first-class remote execution target beside Modal. It is not treated
as a generic SSH machine. A launch must carry the same immutable Aster run contract,
checkpoint lineage, data identity, W&B run identity, cost ceiling and recovery proof
as every other training environment.

The initial provider implementation is deliberately non-spending. It can build and
validate launch plans, but the checked-in `google-credit` profile stays blocked until
the user supplies provider-native credentials and replaces every placeholder.

## Billing and quota gate

Google's current Free Trial documentation states that a non-billable Free Trial
account cannot add GPUs or request quota increases. Activating paid billing unlocks
those capabilities while preserving unused Welcome credit until the original 90-day
expiry. Because usage beyond the credit can then be billed, Aster requires all of:

1. `billing_mode: welcome_credit_upgraded`;
2. confirmed regional GPU quota;
3. a per-job spend ceiling no larger than the Studio policy ceiling;
4. explicit cost confirmation in the immutable contract;
5. `dispatch_enabled: true` only after the preceding facts are verified.

Source: <https://cloud.google.com/free/docs/free-cloud-features>

## Hardware adjudication

Do not assume the most expensive GPU wins. The first controlled matrix uses the same
model, tokens, seed, data order, precision-valid recipe and evaluation budget on:

- `a3-highgpu-1g` / H100 80 GB: primary throughput candidate;
- `a2-ultragpu-1g` / A100 80 GB: price-per-token control;
- `g2-standard-4` / L4 24 GB: small-model cost control.

Rank by validation improvement per wall-clock hour and dollar, with tokens/s, GPU
utilization, VRAM and kernel traces as explanatory metrics. Exact machine type and
zone are recorded. Capacity fallback never merges results as if they came from the
same hardware.

Current GPU-machine reference:
<https://cloud.google.com/compute/docs/gpus>

## Standard before Spot

Standard instances are the first correctness and performance target. Spot becomes an
eligible cost optimization only after exact data/optimizer/RNG resume and a forced
preemption drill pass. Spot can be reclaimed at any time and normally offers only a
short shutdown window; therefore it cannot be used as a substitute for checkpointing.

Source: <https://cloud.google.com/compute/docs/instances/spot>

## Storage and cache topology

The durable layer is a private Cloud Storage bucket:

```text
gs://BUCKET/datasets/                 immutable cleaned/tokenized shards
gs://BUCKET/checkpoints/CONTRACT/     complete hash-verified checkpoints
gs://BUCKET/contracts/                immutable launch contracts
gs://BUCKET/logs/CONTRACT/            stdout, exit record, profiler summaries
```

Each VM uses local disk as a read/cache and training-write layer. The dataset bucket
is mounted read-only through Cloud Storage FUSE; checkpoints are written atomically to
local disk, then copied only after their completion manifest verifies. This avoids
redownloading the whole corpus on each job without pretending object storage has full
POSIX rename semantics. Cloud Storage FUSE file caching/parallel downloads are used
for repeated large reads where measurements show a benefit.

Sources:
<https://cloud.google.com/storage/docs/cloud-storage-fuse/overview>,
<https://cloud.google.com/storage/docs/cloud-storage-fuse/file-caching>

## Secrets and identity

- `gcloud` named configurations are aliases only; account/token values never cross
  the Studio API.
- The VM receives a least-privilege service account.
- Hugging Face and W&B values live in Secret Manager. Contracts contain secret names,
  never values.
- Container images must be Artifact Registry digests built from the recorded Git
  commit. A dirty worktree blocks dispatch.
- The remote bootstrap re-hashes model/train/data configs before execution.

## Recovery contract

Before GCP is promoted, test and retain evidence for:

1. local uninterrupted versus local stop/restart exactness;
2. local to GCP resume;
3. GCP to local resume;
4. GCP standard-instance forced termination and restart;
5. GCP Spot forced preemption after Spot is enabled;
6. identical next-example hash, optimizer/scheduler/RNG state and model result;
7. Hugging Face checkpoint round-trip hash and continuous W&B run history.

The current local exact-resume test is the prerequisite foundation. Credential- and
quota-dependent drills remain blocked until the account is activated.

## Implemented surface

- `configs/providers/gcp_boost.yaml`: non-secret profile and candidate matrix;
- `src/asterlm/cloud/gcp.py`: validation, blocker evaluation, dry-run plan and explicit
  capacity-attempt dispatcher;
- `scripts/gcp_boost.py`: plan-by-default CLI, with a separate `--execute` mutation;
- `scripts/cloud/gcp_startup.sh`: contract fetch, Secret Manager lookup, GCS mount and
  pinned-container execution;
- `scripts/cloud/gcp_shutdown.sh`: bounded container stop and mount cleanup;
- `scripts/cloud/run_contract.py`: provider and input-hash verification before train;
- Studio provider readiness and immutable run-contract support;
- unit tests proving placeholder profiles cannot spend and secrets never enter plans.

## Remaining credential-dependent work

After login/activation: create the project, bucket, Artifact Registry repository and
least-privilege service account; enable APIs; request regional quotas; build the pinned
image; query live prices/quotas; execute a zero/low-cost CPU preflight; then run the
smallest GPU correctness smoke. No architecture-scale spend occurs before those pass.
