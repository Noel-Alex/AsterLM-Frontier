# Research decision ledger

This table prevents the repository from presenting every new paper as a proven improvement.

| Method | Status | Why |
|---|---|---|
| 3:1 KDA3/MLA hybrid | selected laptop backbone | 24 recurrent KDA layers plus 8 local-window MLA anchors; scale frozen on 2026-08-13 |
| Official FLA KDA | baseline CUDA path | maintained optimized kernels |
| PyTorch delta fallback | tests only | correctness/portability, too slow for serious runs |
| MLA-style latent cache | baseline | compact exact content-addressable state |
| q-LoRA | baseline frontier presets | reduces query projection cost/parameters and follows MLA practice |
| Stable LatentMoE, 16 routed top-2 | selected laptop sparse FFN | 1.448B logical / 568.2M active winner; 18+ expert variants failed sustained no-offload fit |
| One shared expert | selected in frozen K3 | always-on common capacity retained while maximizing routed capacity inside the laptop state budget |
| Bias-based load balancing | baseline MoE | avoids large auxiliary-loss pressure |
| Dense 661M control | mandatory | detects MoE undertraining/routing failure |
| MTP | disabled in base pretraining | revisit as a checkpoint-compatible post/base-training experiment after the backbone run is underway |
| Per-head Muon + AdamW | selected primary optimizer | won the completed 4M-token equal-wall screen; AdamW-WSD remains the exact recovery/control recipe |
| QK-Clip / MuonClip | first-class stability candidate | logit stability must include Q/K projection rescaling and per-head logit telemetry |
| WSD | baseline | flexible long-run schedule and final decay |
| BF16 trainable storage | selected laptop training precision | stable Ada-compatible baseline; remote FP8 requires an exact-GPU qualification rather than changing model semantics |
| Transformer Engine FP8 | experiment | supported on Ada; speed/memory shape-dependent |
| Native NVFP4 | unavailable | Blackwell feature, not RTX 4080 hardware |
| TorchAO AdamW4bit/8bit | experiment | directly reduces optimizer-state VRAM |
| APOLLO-Mini | experiment | low-rank optimizer state; external dependency |
| CPU/NVMe optimizer offload | rejected for the laptop run | does not satisfy the user's time-to-quality or no-offload constraint |
| LoQT-style INT4 FFN bases | experiment | trades compute for persistent model memory |
| SSNorm | baseline frontier presets | quantization-friendly outlier control |
| Orthogonal embedding projections | baseline frontier presets | OSP component, foldable at export |
| Hadamard INT4 latent cache | inference experiment | strong memory reduction, must pass retrieval tests |
| Full TurboQuant fused kernels | future kernel work | portable policy implemented, fused kernel not reproduced |
| TorchAO INT4 inference | experiment | reduce model VRAM; quality/speed measured |
| 32K native context | frozen final continuation stage | 2B tokens on qualified high-memory CUDA under the current exact KDA backward path |
| 64K/128K | validation/extension | only claim after retrieval/natural tests |
| 256K | inference target | bounded 8K MLA cache plus recurrent KDA; claim only after retrieval and natural-context gates |
| 1M | inference stretch candidate | parameter-compatible YaRN config exists; no quality claim before full validation |
| DeepSeek-V4 CSA/HCA | Tier-4 candidate | official V4 evidence is strong at 1M; build semantic reference and discover Ada/Hopper/Blackwell crossover separately |
| mHC residuals | Tier-5 candidate | official kernels are SM90/SM100; isolate quality/time-to-loss and build Ada path only after reference parity |
| LongCat zero-compute expert | Tier-3 candidate | potentially relevant to one GPU only when routed tokens physically skip work |
| LongCat shortcut-connected MoE | distributed-only candidate | designed to widen compute/communication overlap; no presumed single-GPU benefit |
| Block AttnRes | excluded from frozen launch model | implemented but insufficiently evidenced; may be studied after the base run starts, not used to reopen launch selection |
| Negative KDA eigenvalues | off by default | possible state tracking gain; stability ablation required |
| EAGLE-3/DeepSpec | integration path | separate draft training and serving system |
| Block-diffusion drafter | deferred | workload-specific and high implementation cost |
| Byte Latent Transformer | separate project | changes tokenizer/encoder/decoder stack entirely |
