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

## Prepared scale frontier

The required scale candidates are now explicit configs rather than extrapolated names. Meta-device
construction verifies the following logical/active parameter geometry on commit `bad468f`:

| Candidate | Total | Active/token | Routed geometry |
| --- | ---: | ---: | --- |
| `aster_k3_latentmoe_868m_a483m` | 868.3M | 483.4M | 12 experts, top-2, 2 shared, 576 latent |
| `aster_k3_latentmoe_1p45b_a568m` | 1.448B | 568.2M | 16 experts, top-2, 1 shared, 704 latent |
| `aster_k3_latentmoe_1p95b_a766m` | 1.954B | 765.9M | 18 experts, top-3, 1 shared, 832 latent |

Each retains Stable LatentMoE, SiTU-GLU, Quantile Balancing and a 3:1 KDA/MLA pattern with
128-dimensional KDA heads. They remain future scale-up tiers. They are not allowed to delay the
frozen laptop launch candidate or restart broad architecture screening.

## Ada utilization diagnosis and GPU-resident grouped backend

A clean, source-pinned three-repetition matrix on commit `e62f134` used sequence length 2048,
microbatch 2, accumulation 8, five warm-up steps and twenty measured updates per repetition. The
reference Stable LatentMoE path reached a median 4,624.9 tok/s at 56% median GPU utilization. The
cached CUTLASS grouped path reached 7,004.0 tok/s at 53% median utilization and 5.24 GiB peak
allocated VRAM. Grouped expert GEMMs therefore removed substantial wall time, but did not resolve
the utilization collapse.

The full PyTorch trace identified 1,029 CUDA stream synchronizations in one profiled APOLLO-Mini
update. Of these, 706 were scalar synchronizations in the optimizer's inherited AdamW step and 276
were associated with fixed-cardinality routing histograms and host expert-size metadata. Switching
the systems probe to fused AdamW reduced the count to 323 and reduced profiled optimizer wall time
from 357 ms to 41 ms, at the cost of approximately 0.94 GiB additional optimizer memory. Energy was
recorded but did not participate in this decision.

PyTorch 2.13's public differentiable `torch.nn.functional.grouped_mm` supports BF16 on the laptop's
SM89 GPU and accepts cumulative jagged offsets resident on CUDA. Aster now exposes this as the
`torch_grouped` physical backend. Fixed-size expert and quantile histograms use GPU `scatter_add_`
instead of `torch.bincount`, so their known output cardinality no longer requires host scalar
discovery. Forward output, input gradient, every expert parameter gradient, cache invalidation and
checkpoint-key parity pass against the dropless reference path. In the first one-update operator
profile, native grouped MM plus fused AdamW reduced stream synchronizations from 1,029 to 185 and
profiled total forward/backward/optimizer wall time from 1.029 s to 0.750 s.

The subsequent four-repetition source-pinned matrix on commit `e8acb30` selected packed CUTLASS for
the laptop: median throughput was 9,294.9 tok/s versus 9,188.1 tok/s for `torch_grouped`, at the same
5.774 GiB peak allocated memory. `torch_grouped` removed all 9,200 host metadata synchronizations
per trial, but its slower backward path erased the forward benefit. Both backends retained zero
post-update weight-restacking bytes after expert storage packing. CUTLASS is therefore frozen as the
laptop physical backend; `torch_grouped` remains diagnostic kernel work and does not reopen model
selection.
