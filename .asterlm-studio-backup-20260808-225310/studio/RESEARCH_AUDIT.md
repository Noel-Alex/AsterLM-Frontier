# AsterLM Studio — Frontier Research Audit

Reviewed: 2026-08-08

This file records why Studio's Architecture page distinguishes stable
implementation from reference implementation and research candidates.

It is deliberately not a shopping list of paper names. A feature is labelled
"implemented" only when the Aster repository contains a real execution path.

## Stable direction: Kimi Delta Attention + MLA-style global attention

Aster's current hybrid direction is strongly aligned with Kimi Linear:

- 3:1-ish recurrent KDA to global latent-attention layers;
- KDA through Flash Linear Attention when available;
- compressed latent global KV state.

Primary source:

- MoonshotAI/Kimi-Linear
  https://github.com/MoonshotAI/Kimi-Linear
- Paper:
  https://arxiv.org/abs/2510.26692

The Kimi repository reports long-context KV/cache and decode benefits for its
own much larger models. Those numbers are **not** automatically Aster's numbers;
Studio requires Aster's own benchmarks.

## Newer recurrent candidate: Gated DeltaNet-2

Gated DeltaNet-2 decouples channel-wise erase and write gates while retaining
channel-wise decay. Its authors position it as a strict generalization of KDA
and Gated DeltaNet and report strong 1.3B/100B-token results.

Primary source:

- https://github.com/NVlabs/GatedDeltaNet-2
- https://arxiv.org/abs/2605.22791

Important licensing point:

The official repository is under the NVIDIA Source Code License-NC. Studio
therefore treats GDN2 as a **research gap**, not code to silently copy into the
MIT Aster tree.

A clean-room/reimplementation decision should be made deliberately after
checking the license and after the current KDA baseline has real measurements.

## DeepSpec / DSpark / DFlash / Eagle3

DeepSeek released DeepSpec as a full-stack draft-model training and evaluation
framework.

Primary source:

- https://github.com/deepseek-ai/DeepSpec
- DSpark paper:
  https://arxiv.org/abs/2607.05147

The current upstream repository lists DSpark, DFlash and Eagle3 and currently
ships target support/checkpoints around Qwen3 and Gemma families.

Aster currently trains MTP heads and includes an exact greedy self-speculative
reference verifier, but it does **not** have an upstream DeepSpec target adapter,
draft checkpoint, or production DSpark/DFlash verifier.

Therefore:

```text
MTP training                 IMPLEMENTED
exact MTP verifier           REFERENCE
DeepSpec/DSpark acceleration NOT INTEGRATED
```

Studio never collapses those three statements into one checkbox.

## Latent-Condensed Attention (ACL 2026)

LCA operates context condensation directly in MLA's latent space and reports
joint compute/cache reductions for long context.

Primary source:

- https://aclanthology.org/2026.acl-long.1176/

This is architecturally interesting for Aster because Aster already stores
semantic latent vectors and a decoupled RoPE channel.

It is not integrated. It should first be prototyped as a separate inference
experiment and compared against Aster's exact latent-cache decode.

## RoBSA (ACL 2026)

RoBSA is a training-free, blockwise sparse decoding method designed
specifically for MLA.

Primary source:

- https://aclanthology.org/2026.acl-long.46/

This may be a cleaner first sparse-MLA inference experiment than changing the
trained architecture because it targets decoding rather than pretraining.

It is not currently in Aster.

## Native Hybrid Attention (ACL 2026)

NHA combines recurrent long-term slots and short-window tokens inside a unified
attention layer.

Primary source:

- https://aclanthology.org/2026.acl-long.176/

It is conceptually related to Aster's objective but is not the architecture
currently trained by Aster. It remains an ablation/research candidate.

## Math fill source: OpenWebMath

Primary source:

- https://huggingface.co/datasets/open-web-math/open-web-math

The dataset card describes 6.3M documents and approximately 14.7B tokens and
exposes a straightforward `text` field. Studio includes it as a directly
usable optional math source.

## Large math/code fill candidates: NVIDIA Nemotron

Primary sources:

- https://huggingface.co/datasets/nvidia/Nemotron-CC-Math-v1
- https://huggingface.co/datasets/nvidia/Nemotron-CC-Code-v1

The current cards describe:

- Nemotron-CC-Math 4+ at roughly 52B tokens;
- Nemotron-CC-Code v1 at roughly 427.9B tokens.

Both repositories require accepting NVIDIA's data-access terms. Their cards
also contain downstream licensing/redistribution notes because model-generated
content was used in parts of the processing pipeline.

Studio therefore marks them **gated / validate first** and never attempts to
paper over a 401/403 as a transient network retry.

## Aster-specific conclusion

The current Aster architecture already contains a meaningful modern stack:

- KDA fast path through FLA;
- MLA-inspired compressed latent global attention;
- absorbed latent single-token decode;
- quantized hot/cold latent KV cache;
- MTP training;
- sparse routed/shared-expert MoE;
- YaRN long-context scaling;
- low-memory training paths and extensive telemetry.

Its biggest currently visible performance/research gaps are:

1. fused MoE dispatch/grouped GEMM;
2. production speculative decoding rather than full-prefix MTP verification;
3. empirical validation of 64K/128K rather than config-level capability;
4. optional comparison with post-KDA recurrent designs;
5. optional sparse/condensed MLA inference;
6. live validation of optional FP8 / low-bit optimizer combinations on the
   exact local CUDA/PyTorch stack.

Those gaps are exactly what the Studio capability audit and experiment controls
are intended to make measurable rather than aspirational.
