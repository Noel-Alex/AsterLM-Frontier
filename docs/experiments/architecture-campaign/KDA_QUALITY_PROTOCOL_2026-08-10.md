# KDA quality and geometry protocol

Status: KDA3 is the leading long-context **systems** candidate, not yet the selected
quality architecture.

## Evidence that must be held together

The repaired compact KDA3/MLA path scales extremely well on the RTX 4080 Laptop:

| Context | Median tokens/s | Mean / median GPU utilization | Peak allocation |
|---:|---:|---:|---:|
| 4K | 6,851.50 | 97.1% / 100% | 3.652 GiB |
| 8K | 6,393.23 | 99.2% / 100% | 3.893 GiB |
| 16K | 5,151.69 | 99.7% / 100% | 4.820 GiB |

The dense MLA control was approximately tied at 4K but used 7.587 GiB. At 8K its
first compiled step took 168.45 seconds, peaked at 10.990 GiB, and then failed before
a warmed measurement. These are systems results only.

The earlier short proxy quality screens did **not** favor KDA:

- at 4,194,304 tokens, compact KDA3 finished at validation main loss 6.7617 and its
  repeat at 6.7734, versus 6.0469 for dense MLA;
- at 16,777,216 tokens over two seeds, KDA1 finished at 6.1914 versus 5.9922/5.9844
  for dense MLA;
- the KDA3 learning-rate screen was shorter and did not close this gap.

Those experiments are too short and too short-context for a final verdict, but they
are valid negative evidence. Faster kernels do not change the KDA equations and
therefore cannot erase it.

The first source-pinned geometry systems probe used the exact current checkout at
4K, batch 1, AdamW, BF16 and compiled surroundings:

| Geometry | Projection width | Median tokens/s | Mean GPU util. | Peak allocation |
|---|---:|---:|---:|---:|
| 16 x 64, safe gate | 1,024 | 6,818.15 | 95.1% / 99% median | 3.531 GiB |
| 8 x 128, safe gate | 1,024 | 7,004.70 | 97.0% | 3.566 GiB |
| 16 x 128, reference gate | 2,048 | 5,344.52 | 98.7% | 4.602 GiB |

Its first compile step took 103.42 seconds and is excluded from warmed throughput.
The expanded/reference path is 23.7% slower than equal-width 128-head KDA at 4K;
quality must repay that compute. The source-pinned 8 x 128 candidate is 2.7% faster
than the compact 16 x 64 median in this first diagnostic, with essentially identical
peak memory. The compact run contained one transient slow measured step, so repeated
warm profiles remain required before treating the small speed difference as durable.
Crucially, both compact geometries now reach 99--100% median GPU utilization: low
utilization is not an intrinsic KDA limitation on this laptop after the kernel and
host-synchronization repairs.

## Fidelity gap found in the old candidate

The original Aster KDA candidate used the model's MLA geometry: 16 heads x 64
dimensions, a 1,024-wide KDA Q/K/V projection, safe gate enabled, and a -5 gate
lower bound.

The released Kimi Linear base configuration uses a separate KDA geometry with
128-dimensional KDA heads. Its 32 x 128 KDA projection is 4,096 wide for a 2,304
hidden-size model, and its released config does not request the fast safe-gate clamp.
It also retains the 3:1 KDA/global-MLA placement and short-convolution size 4.

Therefore the earlier Aster result tested a compact KDA-inspired hybrid, not a
shape-faithful small-scale reproduction of Kimi Linear. This is a research finding,
not a guarantee that the expanded form wins on a laptop.

## Controlled candidates

1. `compact-h64-w1024-safe`: existing 16 x 64, safe gate, lower bound -5.
2. `statewide-h128-w1024-safe`: 8 x 128, same projection width, double
   recurrent matrix capacity per layer. This isolates head/state geometry.
3. `expanded-h128-w2048-reference-gate`: 16 x 128, expanded projections, unclamped
   reference gate. This tests a more Kimi-like ratio at higher active compute.
4. If candidate 3 wins quality, split its causal factors: expanded-safe versus
   expanded-reference-gate, then projection widths 1.25x/1.5x/2x.

Dense MLA and compact KDA controls retain identical data, tokenizer, initialization
seed, update-token batch, optimizer, schedule, validation set and checkpoint cadence.
Compare equal tokens, equal active FLOPs, equal wall time, and equal total parameters;
no single axis substitutes for the others.

## Quality ladder

### Rejection screen

- at least 16.8M real contiguous tokens, two seeds;
- sequence 2K/4K with identical packed documents;
- validation main loss trajectory, not only final total loss;
- gate/decay distributions, gradient norms and recurrent-state norms;
- abort only for numerical failure, not a temporarily worse early loss.

### Decision screen

- at least 100M unique or low-repeat tokens after recipe selection;
- matched dense and KDA candidates at 4K and 8K curriculum points;
- loss versus tokens, wall time and active FLOPs;
- downstream language, math and code validation;
- checkpoint/restart parity and two seeds for finalists.

### Long-context screen

- train/adapt at native 16K then 32K before claiming those lengths;
- RULER-style retrieval, variable-depth needle, repeated keys, distractors,
  multi-hop retrieval, long-code dependency and natural-document perplexity;
- evaluate 4K, 8K, 16K, 32K, 64K and 128K as distinct points;
- report accuracy and latency/memory together, never cache fit alone;
- compare against full/dense MLA with the largest context that safely runs.

The implemented `scripts/long_context_retrieval.py` gate includes deterministic
exact-key retrieval, repeated-key interference and split two-hop association, with
token-positioned depth, exact prompt-plus-answer length, teacher-forced answer
NLL/greedy recall, real incremental-cache prefill, case-level JSONL durability and
source/checkpoint provenance. This deliberately evaluates base pretraining
checkpoints before instruction tuning. Long-code dependency, richer RULER task
families and natural-document perplexity remain separate required gates; synthetic
retrieval alone cannot promote long context.

## Recipe ablations

- KDA head dimension and projection expansion;
- safe gate and -5 lower bound versus reference gate initialization;
- short convolution on/off and kernel size;
- value expansion and grouped-value geometry;
- negative eigenvalues only as a separately justified state-tracking ablation;
- RMS epsilon and recurrent FP32 parameter policy;
- AdamW and Muon learning-rate/warmup sensitivity;
- KDA/MLA ratios 1:1, 3:1 and 7:1 after geometry is fixed;
- final global-attention placement and output gate;
- residual design, including mHC/AttnRes only after the standard-residual baseline.

## Promotion rule

KDA is promoted only if an optimized geometry reaches or beats dense MLA quality on
the decision screen while preserving its measured long-context systems advantage.
If expanded KDA wins quality but is slower at short context, hardware-aware execution
may use different kernels/batches—not different trained architecture semantics. If no
KDA recipe closes the quality gap, it remains a valuable research branch rather than
being forced into the final model.
