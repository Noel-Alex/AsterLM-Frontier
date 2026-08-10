# Tier 0/1 short-context systems baseline — 2026-08-10

Status: systems evidence only. Synthetic tokens do not provide a quality result and
cannot promote or reject an architecture.

## Controlled setup

- git: `28d950d5b88576326196ddce35c2e938a69f8dce`
- GPU/backend: NVIDIA GeForce RTX 4080 Laptop GPU, `cuda-ada-sm89`
- PyTorch/CUDA runtime: 2.13.0+cu130 / 13.0
- sequence/microbatch/accumulation: 2048 / 1 / 1
- precision: BF16 AMP
- optimizer: fused PyTorch AdamW
- three warm-up steps, eight measured steps
- both candidates: dense SwiGLU, RMSNorm, standard residual, no MTP, BF16 cache
- raw records: `runs/architecture-campaign/tier{0,1}-systems/` (ignored artifact plane)

The KDA/MLA candidate has 424.0M trainable parameters versus 411.6M for pure MLA
(+3.0%), because KDA is not parameter-identical to the replaced attention blocks.
This is a matched architecture systems control, not an equal-parameter scaling-law
result.

## Result

| Metric | Dense MLA | Dense 3:1 KDA/MLA | KDA/MLA delta |
|---|---:|---:|---:|
| median tokens/s | 6,416.60 | 5,004.82 | -22.00% |
| median GPU utilization | 95.5% | 78.0% | -17.5 pp |
| peak allocated VRAM | 3.354 GiB | 3.484 GiB | +3.88% |
| median forward CUDA time | 72.11 ms | 92.07 ms | +27.68% |
| median backward CUDA time | 227.57 ms | 242.60 ms | +6.60% |
| median optimizer CUDA time | 17.20 ms | 18.32 ms | +6.56% |
| median sampled board power (statistics only) | 114.10 W | 105.51 W | not ranked |

## Interpretation and next gate

Pure MLA is the clear short-context systems winner at 2K on this Ada shape. This
also verifies that the environment can sustain greater than 95% GPU utilization;
the historical MoE utilization deficit is not a generic WSL or laptop problem.

This result does **not** reject KDA. Kimi Linear's published advantage concerns
long-context cache and decoding behavior. The required next experiment is a
training/inference length sweep using identical candidates and warm-up policy to
find the Ada crossover. Quality comparisons remain required at controlled token and
wall-time budgets.

Energy is retained only as observational telemetry. It must not influence the
winner, early stopping or the final training decision.
