# Architecture-selection mandate

Status: accepted project policy. This document records the user's 2026-08-10
decisions and supersedes architecture-by-reputation. Do not reopen these choices
without new contradictory measurements or a direct user instruction.

## Objective and workload order

Promote the measured Pareto frontier across validation quality per token, wall hour
and dollar; training throughput; prefill/decode speed; VRAM; energy per token;
long-context quality; stability; and recoverability. Logical FLOPs, nominal sparsity
and activated parameters are explanatory metrics, not promotion criteria.

Inference priorities are:

1. laptop batch-1 latency;
2. laptop throughput;
3. strong long-context retrieval;
4. remote batched throughput;
5. minimum deployable memory.

Maintain separate Pareto fronts for the first, second and fourth workloads. One
backend is not required to win all three.

## Size, tokens and context

Sweep sparse candidates around 0.9B/0.45-0.50B active,
1.5B/0.55-0.65B active and 1.9B/0.70-0.80B active, plus dense/hybrid controls at
similar active compute. Compare equal tokens, approximately equal active FLOPs,
equal wall time and equal dollar spend as distinct experiments.

Use 100B tokens as a ceiling. Gates are tens of millions for smoke tests, about
300M for the first serious screen, 1-3B for finalists and 5-10B for high-confidence
promotion. Continue farther only when scaling curves justify it.

Train mostly at 8K, introduce 16K/32K later, and use a targeted extension for
64K/128K when required. The first release must retrieve strongly through 128K.
One million tokens is a stretch experiment, not a release blocker. Evaluate 4K,
8K, 16K, 32K, 64K and 128K.

## Controlled candidate campaign

- Tier 0: fast dense SwiGLU, standard residual, MLA/full-attention reference,
  AdamW. Diagnose data, H2D, graph breaks, Python launches, fusion, LM head,
  checkpointing, allocation, logging, occupancy and GEMM shapes before blaming an
  architecture.
- Tier 1: dense MLA; dense KDA/MLA; each with and without MTP; then properly tuned
  Muon/AdamW and MuonClip-style stability variants.
- Tier 2: four routed experts top-2; eight top-2; eight plus shared; current grouped
  route. Use no-drop correctness, serious bias-based balancing and physical routing
  optimization. MoE has not passed while it is slower in time-to-quality.
- Tier 3: LongCat-style zero-compute/null expert when it truly skips work. Treat
  shortcut-connected MoE as distributed-only unless a single-GPU benefit is shown.
- Tier 4: DSA, NSA, CSA/HCA and compatible cross-layer index reuse. Each needs a
  semantic reference and a physically sparse kernel. Discover per-backend context
  crossover points and reject candidates that lose retrieval.
- Tier 5: standard residual versus Block Attention Residual versus mHC. Do not stack
  novel residual systems before isolated comparisons.

The present prior is a contest among optimized dense KDA+MLA, optimized dense MLA
and modest-granularity MoE KDA+MLA. It is not a predetermined winner.

## Immutable measurement contract

Use one tokenizer, immutable data shard order, controlled seeds and a common eval
harness. Close cheap decisions use at least three seeds. Report domain losses for
web, reference/knowledge, code, math, multilingual and long-form data; practical
general, math and code benchmarks; calibration; contamination; stability; and
multi-mechanism long-context tests.

The mandatory candidate record is implemented by
`asterlm.experiments.validate_benchmark_record`. A partially collected record may
contain explicit nulls, but promotion/Pareto input must be complete. The most
important derived systems metric is time to a fixed validation loss.

For MoE, additionally record tokens per expert, real versus padded tokens, expert
GEMM M/N/K, sort, gather/scatter, grouped GEMM, router and shared-expert time,
synchronization, kernel count and occupancy.

## Backends and kernels

Use a registry for Ada SM89, Hopper SM90, data-center Blackwell SM100, RTX
Blackwell SM120 and a generic fallback. Optimized execution may change fusion,
precision, layout and cache representation, but not mathematical connectivity,
routing, positions or canonical checkpoint meaning.

The initial exact-match policy is implemented in `asterlm.backends`. Unknown CUDA
capabilities deliberately resolve to `cuda-generic`; they never inherit a newer
architecture's kernels through a broad greater-than comparison. Every run manifest
records the resolved backend and its eligible implementation families.

Promotion order is reference correctness, forward/backward numerical parity,
microbenchmark, full block, full model training and quality. A fast isolated GEMM
does not overrule a slower end-to-end model.

Create a separate repository only when a component has a model-agnostic API,
independent correctness suite, independent benchmarks, permissive license boundary
and plausible reuse outside AsterLM. Until then, keep experimental adapters here.

## Providers, cost and durable state

Modal architecture selection is capped at $30 per authorized Workspace, $90 total
initially and $10 per experiment. Paid overage is forbidden. Compare dollars per
million/one-billion trained tokens across exact pinned L40S, A100-80GB, H100/H200,
optional RTX PRO 6000 and B200/B300 only where the model can exploit them.

Use legitimate Modal Workspace membership and provider-native profiles; never pool
or copy friends' credentials. Persistent volumes should cover datasets, HF cache,
and runs/checkpoints. Preparation is content-addressed, hash-verified and idempotent.

Provider switches happen only at durable checkpoint boundaries. A canonical
checkpoint includes model, optimizer, scheduler, precision/scaler, RNG, data cursor,
tokens, step, accumulation state, architecture, tokenizer/data hashes, git and
environment metadata and W&B run ID. Test local-local, Modal-same, Modal A-B,
local-Modal and Modal-local recovery.

Hugging Face is the canonical durable checkpoint/model plane (private repository
`AsterLM-Frontier-checkpoints`), W&B is the canonical telemetry plane (project
`asterlm-frontier`), and GitHub is the source/config plane. Discover authenticated
identities locally; never put tokens in chat, config or Git.

## Licensing and final-run lock

Production dependencies must be permissively licensed. APOLLO remains an isolated
research-only option. Baseline AdamW; first-class challenger Muon for suitable
matrices plus AdamW for sensitive/non-matrix parameters, with QK stability work;
SOAP is a Modal challenger if its total cost is competitive.

The 27 required final-run gates live in
`configs/experiments/promotion_gates.yaml`. No final full-scale run starts while any
required gate is not passed. The dense fallback remains runnable at all times.
