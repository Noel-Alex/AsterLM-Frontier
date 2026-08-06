# VRAM accounting and fitting strategy

## Persistent memory categories

### Parameters

- BF16: 2 bytes per trainable scalar
- LoQT packed INT4 base: approximately 0.5 bytes/value plus scales
- low-rank LoQT updates: BF16 trainable parameters

### Gradients

Usually BF16/FP32 depending on parameter and optimizer implementation. Gradient accumulation does not multiply parameter-gradient storage.

### Optimizer state

- AdamW normally stores two moments, often FP32
- 8-bit/4-bit optimizer paths reduce moments
- APOLLO reduces state rank for selected matrices
- CPU offload removes most optimizer state from GPU
- Muon stores matrix momentum and performs Newton–Schulz work

### Activations

Scale with sequence length and active layers. Checkpointing recomputes them. Saved-tensor CPU offload further reduces GPU residency at PCIe cost.

### Logits

`[batch, sequence, vocab]` can be huge. AsterLM chunks and checkpoints vocabulary projection/cross-entropy and can avoid returning full logits during training.

### Cache

Training does not use the same cache as autoregressive inference. TurboQuant-style KV compression primarily helps inference; it does not remove backpropagation activations.

## Current model persistence estimates

Reference backend, before optimizer/gradients/activations:

- 670M MoE BF16: 1.25 GiB
- 661M dense BF16: 1.23 GiB
- 890M MoE BF16: 1.66 GiB
- 890M LoQT: about 0.93 GiB
- 1.50B MoE BF16: 2.80 GiB
- 1.90B MoE BF16: 3.54 GiB

These values show why larger logical models are plausible, but do not prove they fit during training.

## Decision threshold

Reserve approximately 0.7–1.0 GiB for CUDA context, libraries, allocator variation and safety. The experiment ranker treats about 11.25 GiB peak allocated as the upper practical bound.

## Why inference context can be large

At 262K tokens, the configured global latent cache estimate for the 1.5B/1.9B presets is approximately 0.26 GiB in the current Hadamard-INT4 hot/cold policy, plus recurrent KDA state and temporary decode buffers. This is possible because:

- only 1/4 of layers retain token state,
- each token stores 96 latent + 32 RoPE values rather than full per-head K/V,
- old latent values are packed to four bits,
- cache is processed in chunks.

Quality at that length is not guaranteed.
