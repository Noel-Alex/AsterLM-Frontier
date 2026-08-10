# Research decision ledger

This table prevents the repository from presenting every new paper as a proven improvement.

| Method | Status | Why |
|---|---|---|
| 3:1 KDA/global hybrid | baseline hypothesis | strong cache advantage with periodic exact retrieval |
| Official FLA KDA | baseline CUDA path | maintained optimized kernels |
| PyTorch delta fallback | tests only | correctness/portability, too slow for serious runs |
| MLA-style latent cache | baseline | compact exact content-addressable state |
| q-LoRA | baseline frontier presets | reduces query projection cost/parameters and follows MLA practice |
| MoE 8/12 experts top-2 | candidate | higher total capacity; must beat dense control |
| Shared expert | candidate baseline | always-on common capacity; ablate if it wastes active compute |
| Bias-based load balancing | baseline MoE | avoids large auxiliary-loss pressure |
| Dense 661M control | mandatory | detects MoE undertraining/routing failure |
| MTP depth 2 | baseline | quality objective and native speculative interface |
| Muon + AdamW | first-class challenger | current Aster implementation/config lost its systems probe; rerun with Moonlight-style update scaling and time-to-loss gates |
| QK-Clip / MuonClip | first-class stability candidate | logit stability must include Q/K projection rescaling and per-head logit telemetry |
| WSD | baseline | flexible long-run schedule and final decay |
| BF16 trainable storage | quality reference | stable Ada-compatible baseline |
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
| Block AttnRes | implemented off by default | needs multi-seed quality evidence |
| Negative KDA eigenvalues | off by default | possible state tracking gain; stability ablation required |
| EAGLE-3/DeepSpec | integration path | separate draft training and serving system |
| Block-diffusion drafter | deferred | workload-specific and high implementation cost |
| Byte Latent Transformer | separate project | changes tokenizer/encoder/decoder stack entirely |
