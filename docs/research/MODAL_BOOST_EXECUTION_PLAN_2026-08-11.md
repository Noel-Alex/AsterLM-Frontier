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

The repository is cloned at the contract's exact 40-character commit while the CUDA/PyTorch base image must be pinned by registry SHA-256 digest. The remote entrypoint verifies `git rev-parse HEAD` before it executes the hashed model, train, and data configs.

Every remote run uses `--remote-durable`: full optimizer/RNG/data-state checkpoints are uploaded to a private Hugging Face repository at every configured save, milestone, and final boundary, and the run fails closed if a promised Hub upload cannot be verified. Cross-provider resume names one `runs/.../checkpoints/...` folder plus repository revision; the worker downloads only that folder and runs `verify_checkpoint` before training starts.

## GPU selection

Capacity attempts are ordered and recorded. B300 is a candidate only for a base image declaring CUDA 13.1 or newer; H200, exact H100 (`H100!`), A100-80GB, and L40S remain measurable fallbacks. This order is not a price/performance verdict. Promotion requires matched tokens, loss, wall time, GPU-hours, total spend, recovery, and hardware identity. Silent H100-to-H100-SXM substitution is prohibited in reproducibility campaigns.

## Remaining credential-time gates

1. Log in each authorized workspace as a distinct Modal profile.
2. Create the named `aster-hf-token` and `aster-wandb-api-key` Secrets in each workspace.
3. Pin and record a CUDA/PyTorch image digest; validate FLA, Triton SiTU, Liger/CUTLASS, BF16, and the intended GPU before enabling dispatch.
4. Populate each dataset Volume once and validate its corpus manifest before the first training contract.
5. Run local-to-Modal, forced-termination, Modal-to-local, and cross-workspace Hub resume gates before spending a full credit allocation.

## Primary provider references

- Secrets: https://modal.com/docs/guide/secrets
- Volumes: https://modal.com/docs/guide/volumes
- GPU selection and fallbacks: https://modal.com/docs/guide/gpu
- Sandboxes: https://modal.com/docs/guide/sandboxes
- Existing registry images: https://modal.com/docs/guide/existing-images
