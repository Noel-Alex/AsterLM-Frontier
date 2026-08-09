# LongCat Sparse Attention notes for AsterLM

Primary reference: *LongCat Sparse Attention: Taming the Lightning via Streaming-aware Hierarchical Cross-Layer Indexing*, arXiv:2608.01662 (v2, 2026-08-04), plus the official LongCat-2.0 release.

## The key systems lesson

A sparse FLOP count is not an efficiency result. LongCat profiles DSA's fine-grained token selection and finds two failure modes that matter directly to a 12 GB / bandwidth-constrained laptop:

1. **Scattered KV gathers.** Dynamic token indices cause non-coalesced memory traffic; the paper reports only ~4.5% of peak HBM bandwidth on its hardware. Training backward is worse because scattered gradient updates can serialize.
2. **Indexer cost.** A Lightning Indexer that scores the whole prefix per query is O(L) at decode and aggregate O(L²) during prefill/training; at 1024K it reaches ~90% of per-layer decode latency in their profile.

Therefore vNext2 deliberately retires Aster's first FlexAttention local/landmark prototype and does **not** bolt a per-layer DSA indexer onto Aster.

## LongCat's three fixes

- **Streaming-Aware Indexing (SI):** reserve part of the sparse budget for contiguous sinks + recent window, keeping dynamic retrieval for the remainder. This makes memory access predictable/coalesced.
- **Cross-Layer Indexing (CLI):** one owner layer indexes, neighboring layers reuse it. Naive sharing hurts quality; LongCat uses cross-layer distillation. LongCat-2.0 shares an index every two target layers.
- **Hierarchical Indexing (HI):** coarse page/block retrieval then fine token scoring inside recalled pages, reducing selection complexity from O(L) toward O(L/P + M P). This is training-free at inference.

Their MTP implementation also amortizes index work: all three draft steps share one index pass.

## Aster implications

Aster already has a natural compression unit: the MLA latent KV. Any future sparse path should operate on the **absorbed latent representation**, not reconstruct full per-head K/V just to sparsify it.

The most promising Aster direction after backbone selection is:

1. exact absorbed MLA baseline;
2. contiguous sinks + sliding region;
3. coarse latent-page summaries;
4. sparse fine selection only within a recalled page set;
5. reuse index decisions across adjacent global-attention layers when quality permits;
6. distill shared indices against a dense/absorbed teacher during long-context continuation;
7. benchmark the indexer itself, gathers, top-k, attention kernel and end-to-end layer separately.

## Why vNext2 uses an NSA surrogate scout only

The pinned MIT `fla-org/native-sparse-attention` selected kernel currently requires the GQA query/key head ratio to be a multiple of 16. Aster frontier has 18 query heads and one absorbed latent KV head, so it is not directly shape-compatible. vNext2 benchmarks a clearly labeled 16-query-head surrogate at Aster's absorbed widths (128 Q/K, 96 latent V) to learn whether the kernel family is promising on SM89. It does not use that surrogate as a model-quality conclusion and does not redesign Aster's head count just to satisfy one kernel.

NVIDIA's current cuDNN NSA implementation is Blackwell-focused, so it is also not a drop-in path for SM89 Ada.
