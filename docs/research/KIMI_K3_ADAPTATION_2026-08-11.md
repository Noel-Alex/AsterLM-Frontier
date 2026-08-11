# Kimi K3 adaptation ledger

## Source and scope

The implementation is derived from the official Kimi K3 technical report, inspected from the
source-pinned `MoonshotAI/Kimi-K3` repository on 2026-08-11. Aster keeps every component ablatable;
the report's 2.8T-scale result is evidence that the recipe deserves testing, not evidence that its
exact ratios are optimal on an RTX 4080 Laptop.

## Faithful components now represented

- Normalized LatentMoE: RMSNorm is applied to the weighted routed latent aggregate immediately
  before the shared latent-to-model-width up-projection.
- SiTU-GLU: both routed and shared experts use the report's independently bounded branches with
  beta_gate=4 and beta_up=25. The CUTLASS grouped expert path evaluates the same activation between
  its grouped gate/up and down GEMMs.
- Quantile Balancing: selection uses biased sigmoid affinity, mixture weights remain unbiased, the
  Top-(k+1) cutoff supplies per-token margins, and a vectorized per-expert histogram estimates the
  (1-k/n) quantile. The centered bias is applied on the next optimizer step. Distributed runs
  all-reduce histogram counts rather than gathering token-by-expert margins.
- Per-head Muon: Q/K/V momentum matrices can be split along their output-head dimension and each
  block is orthogonalized separately. This is an explicit training option, not silently applied to
  AdamW controls.
- Block AttnRes: the scaled composite candidate uses eight depth blocks (three layers per block for
  the 24-layer proxy), matching the report's approximately eight-block design principle.

## Initial executable gates

The 220M-class scaled Stable LatentMoE candidate has 269,677,164 total parameters and an estimated
188,272,620 active parameters per token. This makes it an active-compute-matched proxy for the
192,143,616-parameter dense control while exposing additional sparse capacity. Both reference
dispatch and the Ada CUTLASS grouped path completed a two-update forward/backward/evaluation smoke.
These are correctness gates only; the FP8 CUTLASS and BF16 reference losses are not expected to be
bit-identical.

## Required evidence before promotion

1. Matched BF16/FP8 execution profiles with routed rows per expert, grouped-GEMM occupancy,
   optimizer time, routing histogram overhead, VRAM and wall time.
2. AdamW component ablations: normalized latent only, +SiTU-GLU, +Quantile Balancing, then AttnRes.
3. Independently tuned AdamW versus full-matrix Muon versus per-head Muon, and independently tuned
   cosine versus WSD schedules. Reusing one schedule's learning rate and batch size is invalid.
4. Equal-token, equal-active-FLOP and equal-wall-time learning curves over multiple seeds.
5. Long-context retrieval/interference tests before extrapolating K3's one-million-token claims.
6. Scaling the winning recipe through the 0.9B, 1.5B and 1.9B total-parameter tiers, with the final
   laptop target selected from measured fit, throughput and quality rather than proxy size.
