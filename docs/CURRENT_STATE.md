# AsterLM Frontier current state

**Authoritative snapshot: 2026-08-13.** This file owns current status. Dated
handoffs and historical findings remain evidence but do not override it.

## Selected pretraining incumbent

| Property | Frozen value |
|---|---:|
| Logical parameters | 1,448,120,880 |
| Active parameters/token | 568,155,376 |
| Layers / hidden width | 32 / 1,280 |
| Mixer pattern | 24 KDA + 8 latent attention (3:1) |
| KDA geometry | 10 heads x 128; explicit FLA backend |
| Latent attention | 20 heads x 64, latent rank 96, q-LoRA rank 512 |
| MoE | 16 routed, top-2, one shared; expert width 960; latent width 704 |
| Routing | sigmoid scores, quantile bias balancing, z-loss 1e-5 |
| Activation | fused SiTU-GLU |
| Laptop compute | BF16 AMP, packed CUTLASS grouped experts |
| Optimizer | Muon/AdamW, per-head Muon, GPU blockwise-INT8 state |
| Offload | none |
| Base RoPE | YaRN factor 32 from 8K original positions |

The selected 4K full-gradient gate executed gradients across all 1,710 trainable
tensors and every mixer. It measured 2,016.51 tok/s, 97% median GPU utilization,
and 8.065 GiB peak allocated memory on the RTX 4080 Laptop GPU. The earlier
50-60% MoE measurements remain valuable negative evidence, but they are not the
selected geometry's utilization.

## Attention and long context

KDA is Kimi Delta Attention. It is not DeepSeek CSA/HCA. The production base
model pins FLA rather than `auto`, so final training cannot silently fall back to
the slow Torch oracle.

The provisional long-context form keeps global recurrent KDA state and bounds
each of eight latent-attention layers to an 8K compressed history plus 64 sink
tokens. Training is planned at 4K, 8K, 16K, then 32K. A 256K inference target and
1M stretch configuration are parameter-compatible, but neither is a quality
claim before trained-checkpoint retrieval, natural-context perplexity, stability,
latency, and cache gates pass.

CSA/HCA, mHC, sparse gather attention, GDN2, AttnRes, and MTP exist as research
paths. They enter production only through matched learning and deployment tests.

## Scientific status

The incumbent has earned fit, gradient coverage, utilization, kernel parity,
runtime, recovery, and one-seed scale-quality evidence. The final two-seed
production-recipe scale campaign is in progress. Its gate requires the 1.448B
arm to beat the 868M arm separately for each seed at equal wall time, equal tokens,
and equal active FLOPs. A failure is a real rejection signal and must not be
papered over.

The old 220M K3-vs-dense result remains recorded: K3 won slightly per token but
lost equal wall time. Backend and scale have changed since then. Before declaring
architecture selection closed, run two bounded falsification tracks:

1. KDA/KDA/KDA/MLA versus GDN2/GDN2/GDN2/MLA with the same Stable LatentMoE body.
2. The selected sparse model versus a strong dense model under each model's best
   deployable recipe and actual time-to-quality.

These are narrow challengers, not permission to reopen an unbounded architecture
search.

## Data status

The local raw corpus is approximately 87.032B tokens, including about 3B Nemotron
Math tokens beyond the 84.032B primary pool. NVIDIA code shards are still gated
and absent. A raw-token count does not satisfy the final contract.

The source/access audit and directly materialized permissive fallback are recorded
in [`data/CODE_TRANCHE_DECISION_2026-08-13.md`](data/CODE_TRANCHE_DECISION_2026-08-13.md).

The cleaning/promotion path is fail-closed and requires:

- explicit source identities and weights;
- FIM only on declared code sources;
- normalization, quality and secret/PII filters;
- exact and near deduplication across all sources;
- benchmark decontamination and disjoint validation;
- source cleaning reports and a 10K-record audit when enough records exist;
- zero accepted PII findings;
- exact hashes for every sealed artifact;
- at least 50B unique clean tokens, keeping a 100B campaign below 2x replay.

After sealing the corpus, train and seal the 32,768-token tokenizer against that
exact manifest. `correctness_and_data_quality_clear` remains blocked until the
importer verifies all of this.

## 100B curriculum

| Stage | Tokens | Context | Parameter policy |
|---|---:|---:|---|
| Stage 1 | 92B | 4K | all parameters |
| Stage 2 | 3B | 8K | context-extension policy, evidence-gated |
| Stage 3 | 3B | 16K | context-extension policy, evidence-gated |
| Stage 4 | 2B | 32K | context-extension policy, qualified high-memory GPU |

18.4B and 50B are scientific review points. The campaign supervisor must run
checkpoint evaluation and promote the next stage automatically only after its
required gates pass; a YAML transition alone is insufficient.

## Checkpoints and telemetry

The public checkpoint repository is
`philoweeb/AsterLM-Frontier-100B`. A successful round trip already proved exact
Hub byte hashes and W&B history resume. Final checkpoints include model,
optimizer, scheduler, RNG, step/token counters, exact data cursor, configs,
runtime/source manifests, and checksums.

Checkpoint retention is dense near the training head and progressively sparse
in history. Cloud recovery targets roughly five minutes. A single bounded Hub
worker overlaps upload with training, but `latest` advances only after exact
remote hash verification. Both local and remote rolling history keep six recent
points plus eight logarithmically older points per stage; permanent token
milestones and finals are never removed by rolling retention. The pessimistic
four-stage projection is approximately 504 GiB (168 GiB permanent plus 336 GiB
rolling), below the original 0.7 TB planning envelope. Local storage is capped
at 150 GiB; Hub storage warns at 7.0 TB and hard-stops at 7.5 TB. Resume selects
the newest complete compatible checkpoint across local and Hub state.

Telemetry includes loss components, learning rate, gradients and clipping,
parameter health, throughput, utilization, VRAM, clocks, temperature, energy,
router/expert balance and specialization, data cursors, wall time, tokens,
estimated FLOPs, evaluation results, checkpoint events, and diagnostics. Energy
is observational only and never an optimization objective.

## Inference

The exact selected architecture passed a random-initialization laptop runtime
canary: all 32 blocks executed, logits were finite, peak allocation was 2.979 GiB,
and the bounded cache was 0.258 MiB for the tiny canary prompt. Those are runtime
facts, not trained-checkpoint throughput or quality. Benchmark serving/export,
quantization, speculative decoding, vLLM/SGLang integration, and long-context
quality after a real checkpoint exists.

## Providers

Modal workspace profiles and GCP contracts are represented without storing
secrets. Corpus caches, source-pinned images, backend autotuning, full-state Hub
resume, spend ceilings, immediate shutdown, and provider-specific evidence are
part of the design. They are not yet operationally qualified.

No paid dispatch is allowed merely because a provider is configured. Bundle all
planned tests into a single deliberate qualification job per provider/GPU,
measure upload/caching/preemption, shut down immediately, and promote only that
specific numerical/backend family.

## Current mandatory blockers

- `equal_wall_clock`: final two-seed production-recipe scale proof not yet imported.
- `correctness_and_data_quality_clear`: final clean corpus/tokenizer not yet sealed.

Before an irreversible Stage 1 start, also complete clean-clone CI, the narrow
GDN2/dense falsification tests, a multi-hour checkpoint/resume/tracking/thermal
canary, and the automatic stage-evaluation supervisor.

## Environment

The validated WSL lock is `requirements/validated-wsl-cu130.txt`: Python 3.12.3,
PyTorch 2.13.0, CUDA runtime 13.0, Triton 3.7.1, FLA 0.5.2. Fedora CUDA 13.1 and
each cloud image are separate targets. The laptop WSL stack uses the native CUDA
allocator with a 128 MiB split bound; expandable segments have repeatedly failed
virtual-memory mappings despite ample reported free VRAM.

## UI and Git

Studio uses the restored cream theme at the fixed port 8765. Nested architecture
campaigns are included in the live run view, while the SQLite research archive
retains the unbounded comparison history. Metrics tails are read from the end of
the append-only file rather than rescanning the full training history. The active
engineering branch is `Noel/frontier-hardening`; compact promoted evidence lives
in Git so a clean clone can verify it without ignored raw run directories.
Negative findings and invalidated attempts are never erased from the research
record.
