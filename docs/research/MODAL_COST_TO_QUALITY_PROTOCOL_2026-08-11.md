# Modal cost-to-quality protocol (2026-08-11)

## Non-negotiable guardrails

- No paid run occurs until the CUDA image digest, dataset manifest, Secret names, Modal spend budget, exact GPU, timeout, and declared maximum spend all pass the launch-plan blockers.
- One Sandbox runs at a time. Hardware fallback is disabled; each GPU is a declared treatment.
- Image construction and dependency compilation happen before a GPU contract. Modal layer caching and the profile-scoped compiler/Hugging Face cache are reused thereafter.
- Every Sandbox is tagged with contract, profile, and exact GPU. Capture a Modal billing summary/report before the campaign and after each arm.
- Normal completion exits immediately. A failed/hung arm hits a short hard timeout; the operator can use emergency terminate. Longer learning runs use graceful stop and verified full-state Hub upload.
- The large corpus is not uploaded for this screen. A small immutable source-pinned proxy already used by the local campaign is sufficient. Final cleaned corpus artifacts are uploaded only once after their decision-grade manifest is complete.

## Published rate envelope

Rates below are copied from Modal's pricing page on 2026-08-11 and must be refreshed before execution. The maximum includes a conservative 50% allowance for billed CPU and memory. Actual tagged billing, not this estimate, is the research result.

| Exact GPU | Published GPU USD/s | Five-minute guarded maximum |
|---|---:|---:|
| L40S | 0.000542 | $0.2439 |
| A100-80GB | 0.000694 | $0.3123 |
| H100! | 0.001097 | $0.4937 |
| H200 | 0.001261 | $0.5675 |
| B300 | 0.001972 | $0.8874 |

Running all five maximum windows sequentially would be guarded at $2.5047, but the default protocol does not assume all five are necessary.

## Funnel

1. **Zero-GPU build gate:** build the digest-pinned CUDA 13.1 image and verify the exact Git checkout. No training Sandbox.
2. **Cheapest correctness gate:** at most two minutes on L40S. Check imports, CUDA/toolchain identity, FLA KDA, Triton SiTU, Liger/CUTLASS, BF16 numerical parity, checkpoint write, Hub round-trip, graceful stop, and automatic Sandbox exit.
3. **Architecture systems screen:** run the same source-pinned K3 workload for the same measured tokens on L40S, exact H100, and B300, sequentially. Exclude image build, kernel compilation, evaluation, and checkpoint upload from steady-state throughput, but retain them as end-to-end billed time.
4. **Conditional middle candidates:** test A100-80GB or H200 only when capacity, memory fit, kernel support, or the first three results leave a plausible cost/time frontier. Do not spend merely to fill a table.
5. **Short quality confirmation:** only Pareto-frontier GPUs run the same initialization/data order to a fixed validation-loss target. Record tokens, wall time, tagged actual spend, optimizer time, utilization, TFLOP estimate, VRAM, stability diagnostics, and completed-run survival.

## Decision outputs

- **Fastest wall-clock to fixed quality** for boost mode.
- **Most training tokens or quality improvement per dollar** when conserving credits extends total data seen.
- **Largest validated native model/context fit** when additional VRAM changes architecture quality rather than merely batch size.
- **Kernel portability cost:** compilation failures, fallback paths, and performance gaps by architecture.

B300 wins only if its measured time-to-quality or larger-fit advantage justifies its higher tagged cost. L40S wins only if its lower rate offsets slower convergence. GPU utilization is diagnostic; billed wall-clock time to matched quality is decisive.

Primary sources:

- https://modal.com/pricing
- https://modal.com/docs/guide/gpu
- https://modal.com/docs/guide/billing
- https://modal.com/docs/guide/gpu-metrics
