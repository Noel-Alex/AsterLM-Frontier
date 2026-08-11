# Frontier architecture refresh — 2026-08-10

This is a primary-source evidence ledger, not a list of automatic defaults.
Upstream claims remain claims until reproduced at AsterLM scale, context and
hardware.

## Confirmed high-priority candidates

### DeepSeek V4: CSA/HCA + mHC + Muon

DeepSeek's official V4 model card and technical report confirm a hybrid of
Compressed Sparse Attention (CSA) and Heavily Compressed Attention (HCA), mHC
residual connections and the Muon optimizer. At one-million-token context,
DeepSeek reports V4-Pro at 27% of V3.2 single-token inference FLOPs and 10% of its
KV cache. The architecture alternates compressed sparse and heavily compressed
dense long-range paths while retaining a local uncompressed window.

Source: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro

The official source is locally pinned at revision
`b5968e9190ef611bbf34a7229255be88a0e937c1` under
`C:\Users\noela\.cache\aster-research\DeepSeek-V4`. Only the 78.1 KiB
config/inference reference was downloaded; no model-weight shard was requested.
The released configuration makes the research gate concrete:

- every layer keeps a 128-token uncompressed local window;
- CSA uses compression ratio 4 plus a learned 64-head, 128-dimensional indexer
  selecting at most 1,024 compressed positions;
- HCA uses compression ratio 128 and densely attends to its much smaller compressed
  history;
- compression ratios alternate 4/128 after two initial ratio-128 layers;
- mHC uses four residual streams and 20 Sinkhorn iterations.

At 2K sequence length CSA's `min(topk, sequence/ratio)` selects all 512 compressed
positions, so a 2K proxy cannot test sparse retrieval or the claimed long-context
crossover. Aster's semantic and optimized gates must therefore include 8K, 16K and
32K before any conclusion. The highest-value composition candidate replaces K3's
periodic global MLA layers with an ablatable CSA/HCA alternation while retaining KDA
in the recurrent layers; this is an Aster hypothesis, not a published Kimi or
DeepSeek architecture.

This changes AsterLM's status from “deferred mixer” to “Tier-4 research candidate.”
It does not make V4's exact settings a laptop default. The reported comparison is
at 1M and at vastly larger model scale; our principal pretraining length is 8K.

DeepSeek's official TileKernels repository is MIT licensed and contains MoE routing,
quantization and mHC kernels. Its documented requirements are PyTorch 2.10+,
TileLang 0.1.9+, CUDA 13.1+ and NVIDIA SM90/SM100. It therefore supplies algorithms,
references and Hopper/Blackwell candidates, but no documented SM89 Ada path.

Source: https://github.com/deepseek-ai/TileKernels

Action:

1. implement tiny semantic CSA/HCA and mHC references with forward/backward tests;
2. reproduce compression, local branch, causal visibility and indexer semantics;
3. benchmark dense reference crossover by sequence length;
4. evaluate an Ada-capable Triton/TileLang/CUDA route only after the reference and
   retrieval tests pass;
5. reuse upstream SM90/SM100 kernels through a backend adapter where parity and full
   model measurements pass.

### Kimi Linear: 3:1 KDA/MLA

Moonshot's official paper reports that its 48B-total/3B-active 3:1 KDA-to-MLA hybrid
beats full MLA under its controlled recipe, reduces KV cache by up to 75% and reaches
up to 6x decode throughput at 1M. The official repository additionally reports a
3.98x RULER speedup at 128K and 6.3x TPOT at 1M. The paper and repositories are MIT
licensed; KDA is maintained in the MIT-licensed FLA project.

Sources:

- https://arxiv.org/abs/2510.26692
- https://github.com/MoonshotAI/Kimi-Linear
- https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/kda

Action: keep the official FLA KDA kernel as the first optimized implementation,
preserve the slow PyTorch reference for parity, and measure training plus decode
crossover at 4K-128K on Ada. Test hybrid ratios independently; 3:1 is a strong prior,
not a constant baked into the model format.

### Muon and MuonClip

Moonshot's Moonlight work reports about 2x compute efficiency over AdamW in its
compute-optimal scaling experiments and provides a distributed MIT-licensed Muon
implementation. Kimi K2 introduces MuonClip: after Muon updates, Q/K projection
weights are rescaled to control exploding attention logits. Moonshot reports K2 was
trained for 15.5T tokens without a loss spike.

Sources:

- https://github.com/MoonshotAI/Moonlight
- https://arxiv.org/abs/2507.20534
- https://www.kimi.com/blog/kimi-k2

The previous AsterLM Muon result rejects only the current implementation/config: it
was slower and used more memory in that systems probe. It is not evidence against a
proper update-scale, parameter-group and QK-Clip campaign. Promotion must use
time-to-fixed-loss, not optimizer-step time alone.

### LongCat: zero-computation expert and ScMoE

The official LongCat report confirms dynamic zero-computation experts and
shortcut-connected MoE. The former can reduce actual single-GPU work if routed
tokens truly skip expert computation. The latter expands computation/communication
overlap at distributed scale and has no presumed laptop benefit without expert
parallel communication.

Sources:

- https://arxiv.org/abs/2509.01322
- https://github.com/meituan-longcat/LongCat-Flash-Chat

Action: test a trainable null route as Tier 3 with explicit realized compute,
selection-rate, stability and quality measurements. Defer ScMoE until multi-GPU
profiling shows a communication bubble it can hide.

## Reproduction rule

No paper headline is transferable evidence by itself. Every candidate must record:

- exact upstream URL, commit and license;
- what scale/context/hardware the upstream claim covers;
- which AsterLM hypothesis it changes;
- semantic deviations required by the small model;
- reference and optimized parity tolerance;
- end-to-end crossover point per backend;
- equal-token, equal-wall, equal-cost and long-context quality results.

If the optimized implementation remains model-agnostic and independently valuable
after these gates, extract it to a dedicated repository with AsterLM consuming a
pinned version. Do not split early and create two unfinished integration surfaces.
