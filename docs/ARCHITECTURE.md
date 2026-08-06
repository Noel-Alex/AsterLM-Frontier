# Architecture specification

## Design objective

Maximize usable model quality and context on one RTX 4080 Laptop GPU with 12 GiB VRAM. Wall-clock time may grow; failure to fit is unacceptable. The design therefore separates:

- total parameter capacity,
- active parameters per token,
- persistent training state,
- activation memory,
- inference weight storage,
- recurrent state,
- token-indexed cache storage.

No single “parameter count” describes all of those.

## Decoder topology

The default mixer cycle is:

```text
KDA → KDA → KDA → latent global attention
```

This repeats through the network. The rationale is:

- KDA supplies constant-size recurrent memory and efficient sequential processing.
- Periodic exact global attention restores content-addressable copying, retrieval and arbitrary token access.
- A small number of global layers drastically reduces token-indexed cache compared with full attention at every layer.

`layer_pattern` can override the cycle for controlled experiments.

## Kimi Delta Attention

The optimized path wraps `fla-core`'s Kimi Delta Attention implementation. It includes:

- chunk training kernel,
- recurrent one-token inference kernel,
- short convolution,
- fine-grained forget gating,
- optional safe-gate bounds,
- optional negative eigenvalues.

A pure-PyTorch delta-rule implementation is included for tests and portability. Its parameterization differs from FLA; checkpoints pin the resolved backend.

## MLA-inspired global attention

Each global layer uses:

1. optional q-LoRA down-projection,
2. query latent normalization and up-projection,
3. low-rank K/V latent projection,
4. separate small RoPE key/query channel,
5. latent-to-key and latent-to-value up-projections,
6. gated output projection.

During one-token decode, the implementation algebraically absorbs K/V projections and evaluates attention over latent cache chunks. It does not reconstruct a conventional `[tokens, heads, head_dim]` K/V cache.

### Cache width

The frontier presets use:

- latent rank: 96
- RoPE channel: 32
- total raw global state: 128 values per token per global layer

Only approximately one quarter of layers store this state.

### Hot/cold cache

- newest 1K–2K tokens: BF16
- old latent chunks: configurable BF16, FP8, INT8, INT4 or normalized-Hadamard INT4
- RoPE keys: BF16 by default because positional sensitivity may exceed their memory share
- cache processed chunkwise with online softmax
- attention sinks are retained

The Hadamard path is TurboQuant-inspired portable reference code. It should be promoted only if retrieval/perplexity degradation is acceptable.

## MoE

The MoE is intentionally modest rather than an enormous expert bank:

- 8 or 12 routed experts
- top-2 active experts
- one shared expert
- first two layers dense
- sigmoid router scores
- bias-based balancing with optional tiny auxiliary loss and router z-loss
- no deliberate token dropping

The router records expert load and updates a non-gradient routing bias based on load imbalance. The tiny auxiliary loss remains a safety mechanism during early runs, not the primary balancing force.

### Active parameter accounting

The model reports both:

- logical stored parameters,
- active logical parameters per token.

For a top-2 MoE layer, the active count includes two routed experts and shared experts, not every stored expert.

## Dense challenger

A 661M dense model uses the same KDA/global pattern, q-LoRA, cache, MTP, SSNorm and embedding projection. This keeps the MoE comparison focused on expert capacity/routing rather than unrelated architectural changes.

## Outlier-Safe Pre-Training

### Single-Scale RMSNorm

`norm_type: ssnorm` replaces each per-feature RMSNorm scale vector with one learned scalar. RMS statistics are still computed over the hidden dimension. Head-local normalization inside the KDA operator remains per-head because it is part of that recurrence’s internal computation.

### Embedding projection

`embedding_projection: true` adds:

- an orthogonally initialized input-space projection after token lookup,
- an orthogonally initialized output-space projection before the language head.

Both are ordinary dense matrices and are placed on Muon by the hybrid optimizer. At export, the input projection is multiplied into the embedding matrix and the output projection into the language-head matrix. Because the rotations differ, the folded artifact uses untied input/output weights.

## Multi-token prediction

Two low-rank residual future-token heads predict later labels in addition to the main next token. The objective is:

```text
L = L_next + lambda_mtp * mean(L_t+2, L_t+3) + router terms
```

MTP can improve representations and provides proposals for self-speculation. The heads share the final language head and OSP output projection.

## Initialization and stability

- normal weight initialization
- residual-output scaling by `1/sqrt(2 * n_layers)`
- orthogonal OSP embedding projections
- QK-Clip at configurable intervals
- safe KDA gate bound
- FP32 normalization/statistical paths
- router z-loss
- global gradient clipping
- finite-value and norm telemetry

## Context policy

`max_seq_len` is a capacity constraint, not the initial training sequence.

Recommended curriculum:

- 4K/8K general pretraining
- 16K continued pretraining
- 32K dedicated adaptation
- 64K and 128K evaluation/optional continuation only after passing retrieval tests
- 256K is experimental

YaRN is available for extension from an 8K original base. KDA recurrent layers do not hold an ever-growing K/V history, but their finite state can still forget details; long-context evaluation remains mandatory.

## DeepSeek-V4 ideas considered

DeepSeek-V4 combines compressed sparse attention and hybrid compressed attention with local and compressed branches. AsterLM does not force those mechanisms into the baseline because:

- no small single-Ada implementation has yet been proven superior here,
- sparse index selection and fused kernels are substantial engineering projects,
- periodic KDA plus exact latent attention already attacks the dominant cache problem,
- introducing CSA/HCA before the base comparison would confound quality diagnosis.

The architecture is modular enough to add a new mixer kind after the KDA/latent baseline is measured.
