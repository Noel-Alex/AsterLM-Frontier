# Frontier research report and implementation decisions

Updated: 2026-08-05

## Executive conclusion

The previous conservative 220M dense recommendation was not appropriate for the actual goal. The target laptop’s hard constraint is 12 GiB VRAM, while time is flexible. The revised project therefore explores **661M dense through 1.90B MoE**, using compressed optimizer state, activation offload and optional packed INT4 stored FFN/expert matrices.

The current best hypothesis is not “MoE always wins” or “four-bit always wins.” It is:

> A 3:1 KDA/MLA-style hybrid with a modest fine-grained MoE, Outlier-Safe training, MTP and a staged 32K curriculum is the strongest candidate—but it must beat a dense control at equal tokens, and low-bit training variants must match the BF16/FP8 reference closely enough to justify their capacity gain.

## Constraints that shape the design

- NVIDIA RTX 4080 Laptop GPU, 12 GiB VRAM
- single GPU
- slower training acceptable
- model must remain runnable and checkpointable
- inference should be fast but quality has priority
- long context must be useful, not merely configurable
- methods need a realistic implementation path on Ada rather than Blackwell-only hardware

## Architecture research

### Kimi Linear / KDA

**Adopted.**

Kimi Linear introduces Kimi Delta Attention and a hybrid design with recurrent linear-attention layers plus periodic global attention. It targets long-context quality with much smaller state than full attention everywhere. AsterLM uses the official FLA KDA implementation when available and pins the backend in checkpoints.

Adaptation:

- 3:1 KDA/global pattern
- fixed recurrent state in most layers
- safe-gate bound
- short convolution
- recurrent inference kernel
- pure-PyTorch fallback for tests

Reason not to use all-KDA: finite recurrent state may lose exact token-level copying and arbitrary content retrieval. Periodic global layers provide a correction path.

### DeepSeek MLA

**Adopted in adapted form.**

DeepSeek’s MLA principle compresses K/V into a smaller latent representation and reconstructs/absorbs projections during attention. AsterLM implements:

- low-rank latent K/V
- separate RoPE channel
- q-LoRA
- one-token absorbed decoding
- online softmax over latent-cache chunks

This is not a line-for-line copy of DeepSeek’s production kernels. It preserves the key cache-reduction idea in a small, testable implementation.

### DeepSeek MoE

**Adopted with aggressive downscaling.**

Large DeepSeek models show that total capacity and active compute can be decoupled. A laptop still stores every expert, so the implementation uses only 8 or 12 routed experts, top-2, plus one shared expert. Routing uses sigmoid scores and bias-based load balancing with tiny safety losses.

Reasons for caution:

- each expert sees fewer tokens
- expert imbalance is easier with micro-batch 1
- routing/scatter overhead can dominate small matrices
- total expert storage remains resident unless quantized/offloaded

Therefore a dense equal-active-compute challenger is mandatory.

### DeepSeek-V4 CSA/HCA and mHC

**Studied, not forced into the baseline.**

The reported V4 family adds compressed sparse attention/hybrid compressed attention and advanced residual/communication structures. Those mechanisms are promising for enormous context and distributed models, but a fair single-GPU small-model implementation needs:

- sparse selection kernels
- careful index-training objectives
- fused local/compressed branches
- matched small-scale quality studies

The current KDA + exact latent global attention already reduces most cache state while preserving a simple content-addressable path. CSA/HCA is reserved for a later mixer ablation after the baseline is measured.

### Qwen-style hybrid evidence

**Supports the 3:1 prior.**

Sub-billion Qwen hybrid models provide evidence that a 3 recurrent/linear to 1 full-attention pattern can make sense below frontier scale. AsterLM still treats the ratio as an ablation, not a law.

### Mamba and other SSMs

**Deferred.**

Modern SSMs may be competitive, but adding another recurrent family would multiply kernel and stability variables. KDA has official FLA support and directly matches the chosen Kimi hybrid research direction.

## Precision and memory research

### BF16

**Quality reference.**

CUDA parameters are actually stored in BF16, not kept FP32 under autocast. Sensitive statistical paths use FP32. This provides a stable reference loss curve.

### FP8

**Implemented experiment.**

NVIDIA Transformer Engine supports FP8 on Ada. Shape-compatible TE linears and FP8 autocast are available. At this scale, FP8 may or may not improve speed because casting/scaling overhead and matrix dimensions matter; it is measured rather than assumed.

### Native FP4/NVFP4

**Not claimed on RTX 4080.**

NVIDIA’s native NVFP4 training support targets Blackwell. AsterLM does not describe emulated INT4 storage as native four-bit tensor-core training.

### Four-bit optimizer state

**Implemented through TorchAO.**

Optimizer moments are a major persistent-memory cost. AdamW4bit/8bit configurations directly reduce this state. Compatibility and quality must be tested on the exact TorchAO version.

### APOLLO/APOLLO-Mini

**Implemented as optional external optimizer.**

Low-rank optimizer state can fit larger full-rank models. APOLLO-Mini is included in the default memory probes because it preserves trainable weights while compressing optimizer state.

### LoQT-style stored weights

**Implemented experimental path.**

Large FFN/expert matrices are stored as packed INT4 buffers. Trainable low-rank matrices carry updates, which are periodically merged into the base and requantized. Backward reconstructs the base rather than retaining a full BF16 copy.

This is particularly aligned with the constraint “compute is flexible, VRAM is not.” It is not assumed to be quality-neutral.

### Activation offload

**Implemented.**

PyTorch saved-tensor hooks move checkpointed activations to CPU pinned memory. This may be slow over PCIe, but it can make a larger model or longer sequence runnable.

### CPU optimizer offload

**Implemented through TorchAO.**

This is a last-resort/high-value VRAM option for long-context adaptation. It is likely slower, but that matches the user’s priorities.

## Quantization-friendly pretraining

### Outlier-Safe Pre-Training

**Implemented.**

OSP identifies channel-wise scaling and embedding-space concentration as causes of extreme activation outliers. AsterLM adds:

- Muon for hidden matrices
- Single-Scale RMSNorm
- orthogonal input/output embedding projections
- exact projection folding at export
- activation/kurtosis diagnostics

This is important because the intended inference model and cache may use four-bit representations. Quantization robustness should be trained in, not patched in only after pretraining.

### QuaRot/SpinQuant-style rotations

**Partially represented, not fully implemented.**

The cache’s normalized Hadamard transform is related to rotation-based outlier spreading. Full post-training weight/activation rotation and learned SpinQuant matrices are not yet included. OSP is the cleaner first step because it preserves ordinary inference structure after folding.

## Context and cache research

### KDA recurrent state

Most layers avoid token-growing K/V entirely. This is the primary cache reduction.

### Latent cache

The remaining global layers store only latent + RoPE state. This is the secondary reduction.

### TurboQuant-inspired cold cache

Old latent chunks can be normalized, Hadamard transformed and quantized to INT4 with group scales. Recent state remains BF16. This is the tertiary reduction.

The order matters:

```text
fewer cached layers → smaller state per cached layer → lower bits for old state
```

Applying a low-bit method to a conventional full K/V cache would leave much more memory on the table.

### Useful versus nominal context

The model is trained in stages to 32K. 64K/128K are evaluation and optional adaptation targets. No result should advertise 128K solely because RoPE and cache arrays accept it.

## Optimizer research

### Muon and QK-Clip

**Implemented.**

Muon is applied only to ordinary dense matrices, not embeddings, norms, routers, biases, convolution kernels or recurrence constants. OSP embedding projections are an explicit exception and are placed on Muon. QK-Clip periodically rescales query/key projection weights when observed logits exceed a threshold.

### AdamW

Used for sensitive/non-matrix parameters and tied embeddings/head. No weight decay is applied to embeddings, norms, biases, routers and recurrence constants.

### Schedule

Warmup–Stable–Decay is default because it supports long stable training and controlled final decay. Its horizon uses the actual token budget.

## Data research

### General corpus

FineWeb-Edu-Dedup and DCLM provide broad coverage. Cosmopedia contributes structured educational exposition. FineMath adds mathematical density. Stack-Edu adds code while preserving provenance and permissive metadata.

### Token budget

The full local corpus target is 18.4B raw tokens. This is much smaller than frontier-lab training corpora but large enough to make data decisions and expert starvation meaningful. The 1.9B model may be undertrained at this budget; the dense/MoE ablation will reveal that.

### Cleaning

Exact/near dedup, secret/PII handling, benchmark decontamination and source audits are first-class code, not a footnote.

### Post-training

SmolTalk/SmolTalk2/OpenThoughts/UltraFeedback support instruction, reasoning and preference stages. The project avoids treating a giant reasoning trace dump as automatically high quality.

## Speculative decoding research

### MTP

**Implemented and trained jointly.**

It adds little storage and can improve training representations. Exact greedy verification is included.

### EAGLE-3/DeepSpec

**Integration contract included, complete production stack deferred.**

DeepSpec/SpecForge-style systems require draft training, feature extraction, large target-cache generation and fused serving. The repository can export the required target metadata but does not pretend the external system is a small helper function.

### Block diffusion

**Deferred.**

Potentially valuable for predictable code/math blocks, but adds another model/objective and verification strategy. It should be compared only after MTP/EAGLE baselines.

## Final experimental decision tree

1. Validate data sources and start materialization.
2. Run quick VRAM matrix.
3. Select fitting candidates.
4. Train dense/MoE/LoQT candidates for equal pilot tokens.
5. Reject methods with unstable loss or poor expert balance.
6. Run cache quantization and OSP ablations.
7. Promote one base architecture.
8. Train 4K/8K general stage.
9. Adapt 16K then 32K.
10. SFT/DPO.
11. Fold OSP projections.
12. Evaluate BF16/INT4 weights, cache modes and MTP serving.

## Primary sources

- Kimi Linear and KDA: https://arxiv.org/abs/2510.26692
- Flash Linear Attention: https://github.com/fla-org/flash-linear-attention
- DeepSeek-V3: https://arxiv.org/abs/2412.19437
- DeepSeek-V4 Transformers documentation: https://huggingface.co/docs/transformers/model_doc/deepseek_v4
- Kimi K2 / MuonClip: https://arxiv.org/abs/2507.20534
- Outlier-Safe Pre-Training: https://aclanthology.org/2025.acl-long.618/
- OSP implementation notes: https://github.com/dmis-lab/Outlier-Safe-Pre-Training
- LoQT: https://arxiv.org/abs/2405.16528
- APOLLO: https://arxiv.org/abs/2412.05270
- SmolLM3 training recipe: https://huggingface.co/blog/smollm3
- NVIDIA Transformer Engine: https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/
- TorchAO: https://github.com/pytorch/ao
- DeepSpec: https://github.com/hao-ai-lab/DeepSpec
