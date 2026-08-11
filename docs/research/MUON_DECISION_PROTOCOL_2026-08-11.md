# Muon decision protocol for AsterLM

## Correct interpretation of the old slowdown

The 2026-08-10 systems screen did **not** establish that Muon learns worse or that it should be
excluded from AsterLM. It compared the then-current Muon/fused-AdamW hybrid against APOLLO-Mini on
one grouped-MoE workload. Muon delivered 9,612.86 tok/s versus 10,636.84 tok/s for APOLLO-Mini, and
its measured optimizer phase was 460.09 ms versus 216.83 ms. This rejects that implementation and
recipe as the immediate throughput winner; it does not answer final loss, loss-area-under-curve,
time-to-target-loss, stability, or the larger model that each optimizer can fit.

The previous short dense optimizer scout is also not decision-grade. Its three Muon learning rates
were run over only 4.19M tokens with one seed, and its notes already required a matched rerun.

## What the primary sources actually support

- Moonlight reports that Muon required weight decay and parameter-shape-aware update scaling to
  scale reliably, then reached comparable performance to AdamW with about 52% of the training FLOPs
  in its reported scaling-law experiments. Its distributed implementation also removes redundant
  optimizer state and communication overhead.
- Kimi K3 uses Muon for matrix parameters, partitions Q/K/V momentum by attention head, applies K2
  weight clipping and Quantile Balancing, and uses a cosine schedule with 1% linear warmup. The K3
  report explicitly tunes schedule recipes independently; its cosine result cannot be transferred
  to Aster by reusing an AdamW/WSD learning rate.
- LongCat-2.0 reports a 35T-token pretraining run without rollback or irrecoverable loss spikes and
  deploys Muon with targeted tensor-parallel, data-parallel-state, and symmetric-matmul-kernel
  optimizations. This is strong evidence that optimized Muon can be production-stable at scale. The
  public report does not isolate Muon as the sole cause of that reliability, so Aster must not turn
  the correlation into a causal claim.

Primary sources:

- https://github.com/MoonshotAI/Moonlight
- https://github.com/MoonshotAI/Kimi-K3
- https://longcat.chat/blog/longcat-2.0/

## Aster implementation correction

The initial K3 per-head implementation executed five Newton--Schulz iterations inside a Python loop
for each Q/K/V head. That is mathematically valid but creates many small GEMM launches on an RTX
4080 Laptop GPU. Aster now reshapes equal-sized head partitions into a leading batch dimension and
orthogonalizes them with batched matrix multiplication. CPU parity proves the batched result matches
independent head-by-head orthogonalization. CUDA time and numerical parity remain mandatory once the
active architecture campaign releases the GPU.

Muon also exposes low-frequency diagnostics, collected only at the configured diagnostic interval:

- global momentum RMS;
- global orthogonalized-update RMS;
- mean applied-update RMS divided by parameter RMS;
- number of matrix tensors updated by Muon.

These join the existing unclipped gradient norms, parameter RMS/max, QK clipping, max attention
logit, routing balance, throughput, GPU utilization, VRAM, power, and wall-clock records.

The campaign analysis and Studio comparison surface now retain these signals instead of reducing
the decision to terminal loss and tokens/s. Per arm they report completed-run survival, non-finite
loss/gradient counts, pre-clip gradient-norm p95/max, clipping frequency, logged upward loss jumps,
parameter-RMS drift, optimizer share of wall time, and Muon's relative update RMS. These are
diagnostics, not a scalar score: clipping and update magnitude have no universally monotonic
"better" direction, while any non-finite event is a hard stability failure.

## Fair executable decision matrix

`configs/experiments/optimizer_quality_campaign.yaml` and
`scripts/run_optimizer_quality_campaign.py` define a source-pinned, interruption-safe tuning
campaign for the K3 Stable LatentMoE proxy. Every arm uses the same named initialization, corpus,
token order, 16,384-token optimizer update, BF16 numerical family, and CUTLASS grouped experts.
Each optimizer family receives its own learning-rate and schedule search:

1. AdamW + WSD;
2. AdamW + cosine;
3. full-matrix Muon/AdamW + cosine;
4. K3 per-head Muon/AdamW + cosine;
5. APOLLO-Mini + WSD as the low-state-memory challenger.

Muon/cosine and AdamW/cosine use a 1% warmup prior; WSD arms receive a separate 5% warmup prior.
These are tuning candidates, not assumed optima. Microbatch/accumulation is held at the established
4x2 utilization geometry for the first screen; a second fit/utilization search is required at the
winning optimizer and native model scale.

The tuning screen uses one seed only to eliminate clearly bad learning rates. No optimizer is
selected from that screen. Family winners proceed to a multi-seed confirmation with:

- final validation loss mean and standard deviation;
- normalized token-curve and wall-curve loss AUC;
- equal-token, equal-active-FLOP and equal-wall loss;
- time and tokens to common target losses;
- optimizer CUDA time, end-to-end tok/s and GPU utilization;
- peak VRAM and the largest native-scale model each optimizer can fit;
- non-finite failures, loss spikes, gradient clipping frequency, QK clipping and max-logit growth.

## Current decision

Muon is a **first-class challenger**, not rejected and not yet selected. The best optimizer for the
final run will be the recipe that minimizes stable wall-clock time to matched quality while fitting
the strongest validated model. A modest per-step cost is acceptable if Muon saves substantially
more steps or prevents an otherwise failed run. Conversely, a stability story alone does not win if
an independently tuned AdamW or APOLLO recipe reaches the same quality materially sooner.
