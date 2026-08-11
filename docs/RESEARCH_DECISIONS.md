# Research decision ledger

This table prevents the repository from presenting every new paper as a proven improvement.

| Method | Status | Why |
|---|---|---|
| 3:1 KDA3/MLA hybrid | selected laptop backbone | best retained long-context/cache design with periodic exact retrieval; frozen in the 2026-08-11 pretraining selection |
| Official FLA KDA | baseline CUDA path | maintained optimized kernels |
| PyTorch delta fallback | tests only | correctness/portability, too slow for serious runs |
| MLA-style latent cache | baseline | compact exact content-addressable state |
| q-LoRA | baseline frontier presets | reduces query projection cost/parameters and follows MLA practice |
| Stable LatentMoE, 8 routed top-2 | selected laptop sparse FFN | small matched-screen quality edge plus practical full-state fit; physical backend remains separately replaceable |
| Two shared experts | selected in frozen K3 | always-on common capacity retained from the quality-screened configuration |
| Bias-based load balancing | baseline MoE | avoids large auxiliary-loss pressure |
| Dense 661M control | mandatory | detects MoE undertraining/routing failure |
| MTP depth 2 | baseline | quality objective and native speculative interface |
| Per-head Muon + AdamW | selected primary optimizer | won the completed 4M-token equal-wall screen; AdamW-WSD remains the exact recovery/control recipe |
| QK-Clip / MuonClip | first-class stability candidate | logit stability must include Q/K projection rescaling and per-head logit telemetry |
| WSD | baseline | flexible long-run schedule and final decay |
| BF16 trainable storage | selected laptop training precision | stable Ada-compatible baseline; remote FP8 requires an exact-GPU qualification rather than changing model semantics |
| Transformer Engine FP8 | experiment | supported on Ada; speed/memory shape-dependent |
| Native NVFP4 | unavailable | Blackwell feature, not RTX 4080 hardware |
| TorchAO AdamW4bit/8bit | experiment | directly reduces optimizer-state VRAM |
| APOLLO-Mini | experiment | low-rank optimizer state; external dependency |
| CPU optimizer offload | long-context fallback | very large VRAM savings, slower PCIe path |
| LoQT-style INT4 FFN bases | experiment | trades compute for persistent model memory |
| SSNorm | baseline frontier presets | quantization-friendly outlier control |
| Orthogonal embedding projections | baseline frontier presets | OSP component, foldable at export |
| Hadamard INT4 latent cache | inference experiment | strong memory reduction, must pass retrieval tests |
| Full TurboQuant fused kernels | future kernel work | portable policy implemented, fused kernel not reproduced |
| TorchAO INT4 inference | experiment | reduce model VRAM; quality/speed measured |
| 32K native context | target | genuinely trained stage |
| 64K/128K | validation/extension | only claim after retrieval/natural tests |
| 256K | stretch | configuration capacity, not current quality claim |
| DeepSeek-V4 CSA/HCA | Tier-4 candidate | official V4 evidence is strong at 1M; build semantic reference and discover Ada/Hopper/Blackwell crossover separately |
| mHC residuals | Tier-5 candidate | official kernels are SM90/SM100; isolate quality/time-to-loss and build Ada path only after reference parity |
| LongCat zero-compute expert | Tier-3 candidate | potentially relevant to one GPU only when routed tokens physically skip work |
| LongCat shortcut-connected MoE | distributed-only candidate | designed to widen compute/communication overlap; no presumed single-GPU benefit |
| Block AttnRes | excluded from frozen launch model | implemented but insufficiently evidenced; may be studied after the base run starts, not used to reopen launch selection |
| Negative KDA eigenvalues | off by default | possible state tracking gain; stability ablation required |
| EAGLE-3/DeepSpec | integration path | separate draft training and serving system |
| Block-diffusion drafter | deferred | workload-specific and high implementation cost |
| Byte Latent Transformer | separate project | changes tokenizer/encoder/decoder stack entirely |
