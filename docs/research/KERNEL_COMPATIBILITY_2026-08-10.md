# AsterLM kernel compatibility - 2026-08-10

This record separates architecture names from the kernels that are actually executable on the
RTX 4080 Laptop (Ada, SM89). A paper or repository being impressive is not evidence that its
fastest kernel is active in an Aster run.

## Current verified paths

| Component | Aster use | SM89 status | Evidence |
|---|---|---|---|
| NVIDIA Transformer Engine 2.17 | FP8 linears and grouped experts | Pass | Delayed/current FP8 and grouped-MoE CUDA forward/backward probes pass |
| FLA 0.5.2 KDA | Training KDA chunk kernels | Pass | FLA CUDA forward/backward probe passes; current implementation is Triton |
| FLA AttnRes | Experimental residual mixing | Pass | CUDA forward/backward probe passes |
| PyTorch fused linear cross-entropy | LM loss | Pass | CUDA forward/backward probe passes |
| Moonshot FlashKDA | Fused KDA inference | Not compatible | Official build supports SM90a+; Aster screen uses head dim 64 while v1 requires K=V=128 |
| DeepSeek FlashMLA | MLA inference kernel | Not compatible | Official repository requires SM90/SM100 |

Machine-readable evidence is stored in `runs/setup/capabilities-wsl-cu13-fixed.json` and bound by
hash from `runs/setup/runtime-manifest-wsl-cu13.json`.

## What Aster currently means by KDA

`src/asterlm/layers/kda.py` instantiates FLA's `KimiDeltaAttention` with chunk mode, short
convolution, fused in-kernel gate operations, `safe_gate=true`, and lower bound `-5`. During
training, FLA 0.5.2 uses its autograd-capable Triton chunk forward/backward implementation.

FLA 0.5.2 also contains a dispatcher for Moonshot FlashKDA, but that backend is used only under
`torch.inference_mode()`, requires the separately installed `flash_kda` extension, BF16,
K=V=128, equal query/value head counts, and safe-gate inputs. The extension is not installed in
the recovered environment because the official build currently lists only SM90a, SM100a,
SM103a, and SM120a. Aster's vNext2 screen uses head dimension 64 and the laptop is SM89, so it
would be rejected on two independent constraints.

Moonshot's design note says its chunk-16 math uses SM80 MMA instructions and is conceptually
portable across modern NVIDIA GPUs. The published build and launch code nevertheless require
SM90 or newer today. Portability of the mathematical instruction path must not be confused with
availability of a tested Ada binary.

## MLA reality

Aster's `attention_train_backend=absorbed_sdpa` is an algebraically absorbed MLA-style path using
PyTorch SDPA. It is not DeepSeek FlashMLA. The earlier A/B/B/A scout found the absorbed path about
4.7% faster than reconstructed MLA for the tested shape; that remains a useful laptop baseline,
not a claim of datacenter FlashMLA performance.

## Decision

Do not attempt to install official FlashKDA or FlashMLA on this SM89 laptop and do not write an
Ada port before profiling the verified FLA/Transformer Engine paths. The next evidence gates are:

1. matched end-to-end training profiles for dense, grouped MoE, and latent MoE;
2. KDA Triton kernel traces at Aster's exact head/sequence shapes;
3. separate prefill and recurrent-decode inference measurements;
4. only if KDA remains a dominant hotspot, estimate an SM89 FlashKDA port against the expected
   benefit and required correctness coverage.

## Primary sources

- Moonshot FlashKDA README and build constraints: https://github.com/MoonshotAI/FlashKDA
- Moonshot FlashKDA design note: https://github.com/MoonshotAI/FlashKDA/blob/master/docs/20260420-flashkda-v1-deep-dive.md
- FLA KDA integration: https://github.com/fla-org/flash-linear-attention
- DeepSeek FlashMLA hardware requirements: https://github.com/deepseek-ai/FlashMLA
