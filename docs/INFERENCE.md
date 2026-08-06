# Inference, context and cache

## Quality-first policy

Inference speed is optimized after preserving model quality. The default order is:

1. use the trained architecture without changing outputs,
2. fold training-only OSP projections,
3. use fused KDA recurrence,
4. use absorbed latent-attention decode,
5. quantify cache compression error,
6. quantize weights only after perplexity/task evaluation,
7. add speculative decoding only after net speed is measured.

## Runtime

```bash
python scripts/infer.py \
  --checkpoint artifacts/aster-osp-folded \
  --prompt "Explain delta-rule memory and attention." \
  --max-new-tokens 256
```

Interactive chat:

```bash
python scripts/chat.py --checkpoint artifacts/aster-osp-folded
```

Supported sampling:

- greedy
- temperature
- top-k
- top-p
- min-p
- repetition penalty

Prefill is chunked so long prompts do not require a single enormous temporary operation.

## Cache architecture

### KDA layers

KDA stores fixed recurrent and short-convolution state. Its memory does not grow linearly with context length.

### Global layers

Each token stores a compact latent and RoPE key. Cache policy is configured independently of training precision:

- `bfloat16`
- `float8`
- `int8`
- `int4`
- `hadamard_int4`

The hot/cold policy keeps recent tokens at higher precision and compresses old chunks.

### Hadamard INT4

The reference implementation:

1. pads to a suitable power-of-two width,
2. applies a normalized fast Walsh–Hadamard transform,
3. performs groupwise symmetric INT4 quantization,
4. stores scales and packed values,
5. dequantizes one chunk at a time during online-softmax attention.

It avoids storing or reconstructing a full conventional K/V cache. It is not yet a fused kernel, so memory can improve before speed does.

## Context claims

Report these separately:

- configured maximum
- maximum prompt that fits
- native trained context
- retrieval-validated context
- natural long-document validated context

The frontier configs may permit 128K or 256K positions, but the initial legitimate claim is **32K native** after the 32K adaptation phase. Extend the claim only after:

- needle retrieval at multiple depths
- repeated-key and distractor tests
- multi-hop retrieval
- long-code dependency tests
- natural-document perplexity
- generation coherence

## Weight quantization

Runtime/export supports TorchAO weight-only INT4/INT8 for safe large FFN/MTP matrices. Attention and recurrence modules are not blindly replaced because custom algebra and fused kernels may depend on direct access to their weights.

Benchmark:

- model size
- first-token latency
- decode tokens/s
- peak VRAM
- perplexity delta
- downstream score delta

A smaller model file is not evidence of faster execution.

## OSP folding

```bash
python scripts/export_osp_merged.py \
  --checkpoint runs/aster-final \
  --output artifacts/aster-osp-folded
```

The exporter verifies shape compatibility and produces a normal model with no embedding-projection operations. Input/output weights become untied because the two learned rotations are distinct.

## Compilation and CUDA graphs

`torch.compile` is optional. Measure cold compilation separately from warmed throughput. Dynamic cache structures and custom FLA/TorchAO tensor subclasses can reduce compilation reliability.

CUDA graphs are a future optimization for stable one-token decode shapes. They should be introduced after the eager cached path is correct and memory-stable.

## Required benchmark table

```text
checkpoint | context | cache dtype | hot tokens | weight dtype |
prefill tok/s | TTFT | decode tok/s | peak alloc | peak reserved |
perplexity delta | needle accuracy | MTP acceptance
```
