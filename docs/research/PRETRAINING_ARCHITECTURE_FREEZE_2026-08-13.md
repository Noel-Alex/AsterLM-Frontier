# Pretraining architecture and scale freeze — 2026-08-13

This is the current launch decision. It supersedes the 270M scale conclusion in
`PRETRAINING_ARCHITECTURE_FREEZE_2026-08-11.md` and resolves the scale reopening
recorded on 2026-08-12. The mechanism evidence from those documents remains part
of the audit trail.

## Selected model

| Property | Frozen value |
|---|---:|
| Logical parameters | 1,448,120,880 |
| Active parameters per token | 568,155,376 |
| Transformer blocks | 32 |
| Hidden width | 1,280 |
| Vocabulary | 32,768 |
| Routed experts | 16 |
| Active routed experts | 2 |
| Shared experts | 1 |
| Expert hidden width | 960 |
| Latent-MoE bottleneck | 704 |
| KDA / MLA layers | 24 / 8 |
| KDA geometry | 10 heads × 128 |
| MLA geometry | 20 heads × 64; latent rank 96; RoPE width 32 |

The parameter source of truth is
`configs/model/aster_k3_latentmoe_1p45b_a568m.yaml`. The model uses the 3:1
KDA3/MLA hybrid, Stable LatentMoE, SiTU-GLU, sigmoid top-2 routing, quantile
load balancing, RMSNorm, tied embeddings, and no MTP objective in base
pretraining. KDA is Kimi Delta Attention; it is not a renamed implementation of
DeepSeek CSA/HCA.

## Why this scale won

The matched 1,048,576-token scale run on source commit `fa88f30` compared the
868M and 1.448B configurations with the same optimizer/execution family. The
1.448B candidate had lower terminal evaluation loss (6.64122 vs 6.68696), lower
equal-wall loss (6.68065 vs 6.68696), and lower equal-active-FLOP loss (6.68476
vs 6.68696). This is short, one-seed evidence, so it is not presented as a
scaling-law proof; it is a directional matched control combined with the
systems frontier.

The larger expert expansions did not pass the sustained 12 GiB laptop gate:
24 experts failed the full 8K update, 20 experts failed after persistent
optimizer state, and 18 experts completed one update but OOMed during the next
backward. CPU/NVMe optimizer or parameter offload was not accepted as a fit
result. The 16-expert, 1.448B model is therefore the largest currently
defensible no-offload laptop architecture, not an arbitrary small proxy.

## Laptop execution evidence and boundary

The final full-parameter 4K gate used BF16 AMP, packed CUTLASS experts, and
GPU-resident INT8 Muon state with exact per-parameter gradient release after
the update. Across two measured post-warm-up steps it achieved:

- 2,016.51 median training tokens/s;
- 97% median GPU utilization;
- 8.065 GiB peak allocated and 8.373 GiB peak reserved memory;
- gradients for all 1,710 trainable tensors and all 32 mixer blocks.

The machine-readable evidence is
`runs/tmp/1p45b-a568m-int8-full-4k-auto-final.json`. A prior complete 8K fit is
retained at `runs/tmp/1p45b-int8-8k-seg2-fit.json`, but repeated current-session
8K attempts encountered WSL/WDDM allocation-budget failures on small allocations.
For that reason the high-volume base stage is 4K; 8K is a separately gated
continuation rather than a promise inferred from nominal free VRAM.

Exact-gradient 32K KDA backward does not currently fit the 12 GiB laptop. The
32K stage is retained and must run on a qualified high-memory CUDA target until
a lower-memory exact KDA kernel passes semantic and gradient parity. Invalid
checkpoint runs that silently omitted mixer gradients were rejected and are not
used as performance evidence.

## Frozen 100B curriculum

| Stage | Tokens | Context | Trainable policy | Intended placement |
|---|---:|---:|---|---|
| 1 | 92B | 4K | all parameters | laptop or qualified cloud |
| 2 | 3B | 8K | context-extension tensors | laptop after a fit canary, or cloud |
| 3 | 3B | 16K | context-extension tensors | qualified high-memory CUDA preferred |
| 4 | 2B | 32K | context-extension tensors | qualified high-memory CUDA required by current path |

Context extension freezes the embedding/head and MoE FFN knowledge bank while
training every mixer, normalization, and routing tensor. The trainer asserts
that every trainable mixer block receives a gradient. This protects learned
knowledge while spending the long-context budget on the subsystems that need it.

## Inference context

The 256K runtime uses the parameter-compatible long-context config. KDA keeps
constant-size recurrent state and each MLA layer keeps an 8K compressed latent
window plus sinks, so cache growth is bounded rather than proportional to the
full 256K history. A parameter-compatible 1M YaRN configuration also exists at
`configs/model/aster_k3_latentmoe_1p45b_a568m_1m_inference.yaml`.

Neither configuration length is a quality claim. The final checkpoints must
pass token-exact retrieval, repeated-key interference, two-hop retrieval,
natural-document perplexity, generation stability, prefill/decode throughput,
and cache-memory gates at the claimed length. Until then 256K is the configured
target and 1M is explicitly a stretch candidate.

## Reopen rule

Do not restart broad architecture searching. Reopen this choice only for
non-finite training, semantic/gradient mismatch, exact-resume failure, failure
of the declared stage placement, or a predeclared matched control that wins
time-to-quality. Backend/kernel improvements may replace execution without
changing checkpoint semantics.
