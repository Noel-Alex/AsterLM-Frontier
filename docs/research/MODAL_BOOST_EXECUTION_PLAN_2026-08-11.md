# Modal boost execution plan (2026-08-11)

## Decision

Modal is an explicitly profiled, asynchronous training backend; it is not a replacement for the local execution engine and it is not allowed to hide architecture or optimizer changes. Each authorized Modal workspace has its own CLI profile, app name, dataset Volume, package/cache Volume, and checkpoint Volume. Credentials remain in Modal's native profile and Secret stores. Aster Studio receives aliases and Secret names only.

The real adapter is `modal_sandbox_v1`. Studio can create an immutable contract, build a dry-run launch plan, and dispatch only after a second cost confirmation. Defaults are deliberately blocked until the profile has a digest-pinned NVIDIA image, matching credentials, and `dispatch_enabled: true`.

Each launch creates exactly one training Sandbox and stops trying GPU fallbacks as soon as that
Sandbox exists. The hard contract/profile timeout is a billing backstop. Normal completion exits
the Sandbox immediately. Studio exposes a graceful stop that waits for an optimizer boundary,
writes full state, verifies the Hub upload, and then exits; an emergency terminate is also present
to stop billing immediately when losing work since the last completed checkpoint is acceptable.

## Remote filesystem contract

| Container path | Durable backing | Purpose |
|---|---|---|
| `/opt/aster/data` | profile-scoped Modal Volume | already-prepared datasets, preserving repository-relative `data/...` paths |
| `/var/cache/aster` | profile-scoped Modal Volume | Hugging Face, compiler, and package caches |
| `/opt/aster/runs` | profile-scoped Modal Volume v2 | live metrics, diagnostics, and atomic checkpoints |

Remote contracts bind the decision-grade clean-corpus manifest hash as an input. The worker verifies
that manifest from the mounted dataset Volume before model allocation, so an empty, stale, raw-only,
or partially uploaded cache fails without consuming a training run. Raw downloaded corpora are not
uploaded merely because they exist locally; the final cleaned/deduplicated/decontaminated artifacts
are staged once per authorized workspace after their immutable manifest is complete.

The submitter now enforces this before `Sandbox.create`: it opens the existing dataset Volume with
`create_if_missing=False`, reads the exact manifest path from the launch contract, and checks its
SHA-256. A missing Volume, missing manifest, or stale manifest stops before image construction and
before any GPU allocation. The mounted cache Volume also persists `HF_HOME`, `TRITON_CACHE_DIR`,
`TORCH_EXTENSIONS_DIR`, and `XDG_CACHE_HOME`, so compiler artifacts and package/model caches survive
short-lived containers.

## One-time clean cache staging

Cache staging is dry-run by default and never creates a Sandbox or requests a GPU:

```bash
python scripts/modal_stage_cache.py \
  --profile noelalex404 \
  --manifest data/clean-frontier/clean_manifest.json \
  --output runs/modal/cache-stage-noelalex404.json
```

Only after reviewing that plan and selecting the owning `MODAL_PROFILE` may `--execute` be added.
Execution rehashes every sealed local artifact, uploads only the data config and artifacts named by
the decision-grade manifest, and writes the manifest in the same Volume v2 batch as the atomic commit
marker. Volume v2 content/block hashes make an interrupted rerun incremental: already-present blocks
are reused. The raw acquisition tree is never inferred or swept into the upload.

## Consolidated qualification

Remote qualification is also dry-run by default. `scripts/modal_qualify.py` builds one exact-GPU,
short-timeout Sandbox plan from `configs/providers/modal_qualification.json`. That single process runs
the CUDA/toolchain, fused-kernel, architecture, optimizer, checkpoint, and exact-resume checks
sequentially against synthetic or repository fixtures. It continuously writes one durable JSON
summary plus per-check logs under `/opt/aster/runs/modal-qualification/<id>/`, then exits immediately;
the process exit tears down the Sandbox. It mounts the persistent compiler cache but does not mount or
download the full corpus. There is no automatic GPU fallback, parallel fleet, or second container.

No qualification or training Sandbox has been launched by this implementation work. Before any paid
execution, the exact checklist, GPU, timeout, worst-case cost, pinned image, and current local evidence
must be reviewed together so all necessary remote-only checks are collected in that one session.

The repository is cloned at the contract's exact 40-character commit while the CUDA/PyTorch base image must be pinned by registry SHA-256 digest. The remote entrypoint verifies `git rev-parse HEAD` before it executes the hashed model, train, and data configs.

Every remote run uses `--remote-durable`: full optimizer/RNG/data-state checkpoints are uploaded to the public Hugging Face repository at every configured save, milestone, and final boundary, and the run fails closed if a promised Hub upload cannot be verified. Cross-provider same-stage resume names one `runs/.../checkpoints/...` folder plus repository revision; the worker downloads only that folder and runs `verify_checkpoint` before training starts. A context-stage transition uses the separate Hub initialization field, which supplies `--init-checkpoint` and intentionally resets optimizer/scheduler state. Studio refuses contracts that specify both modes.

## GPU selection

Every contract must explicitly name exactly one GPU. There is no paid automatic fallback or silent
hardware substitution. B300 is a candidate only for a base image declaring CUDA 13.1 or newer;
H200, exact H100 (`H100!`), A100-80GB, and L40S are separate cost-to-quality treatments.
Promotion requires matched tokens, loss, wall time, GPU-hours, actual spend, recovery, and hardware
identity.

The 2026-08-11 published GPU rates are recorded per candidate and the launch plan computes a
worst-case GPU cost from the hard timeout with a conservative 50% CPU/memory contingency. Dispatch fails closed
when the contract's declared spend is below that guard, when the workspace/environment spend budget
has not been confirmed, or when no exact GPU is selected. Sandbox tags bind subsequent billing
reports to the Aster contract. Rates must be refreshed from https://modal.com/pricing before a paid
campaign.

## Remaining credential-time gates

1. Log in each authorized workspace as a distinct Modal profile.
2. Create the named `aster-hf-token` and `aster-wandb-api-key` Secrets in each workspace.
3. Pin and record a CUDA/PyTorch image digest; validate FLA, Triton SiTU, Liger/CUTLASS, BF16, and the intended GPU before enabling dispatch.
4. Populate each dataset Volume once with `modal_stage_cache.py` and validate its corpus manifest before the first training contract.
5. Run local-to-Modal, forced-termination, Modal-to-local, and cross-workspace Hub resume gates before spending a full credit allocation.

## Primary provider references

- Secrets: https://modal.com/docs/guide/secrets
- Volumes: https://modal.com/docs/guide/volumes
- GPU selection and fallbacks: https://modal.com/docs/guide/gpu
- Sandboxes: https://modal.com/docs/guide/sandboxes
- Existing registry images: https://modal.com/docs/guide/existing-images
