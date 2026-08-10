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
| DeepSeek DeepGEMM | Dense/grouped FP8 GEMM and Mega-MoE | Not compatible | Current upstream requires SM90 or SM100; laptop is SM89 |
| DeepSeek DeepEP | Expert-parallel dispatch/combine | Not applicable locally | Targets multi-GPU expert parallel communication, not a one-GPU dispatch loop |
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

## MoE grouped-kernel reality

`src/asterlm/layers/moe_grouped_te.py` already follows NVIDIA's established grouped-expert pattern:
it packs token assignments, provides on-device per-expert split sizes, and runs
`GroupedLinear -> SwiGLU -> GroupedLinear` through Transformer Engine's operations API. This is not
the slow Python-per-expert reference path. NVIDIA's Mixtral tutorial explains why grouped GEMM helps
but also explicitly notes that small expert GEMMs may still be too small to fill tensor cores.

Two newer Transformer Engine capabilities must not be conflated with what is active here:

- TE 2.14+ can fuse the entire grouped GEMM + activation + grouped GEMM pipeline for MXFP8/NVFP4.
  The installed TE 2.17 fuser source enables that joint kernel only for those block-scaled recipes;
  Aster's current systems profile uses delayed-scaling FP8, so it receives grouped GEMMs but not that
  new joint block-scaled MLP kernel. Ada support and numerical behavior need an executable probe before
  considering a recipe change.
- TE can store all expert weights in a single grouped parameter, reducing Python/optimizer overhead.
  Aster deliberately retains individual authoritative expert parameters for checkpoint compatibility,
  which preserves existing names but leaves optimizer tensor fragmentation visible in the trace.

The always-on shared expert currently executes as an ordinary TE `SwiGLU` after the routed grouped
call. NVIDIA Megatron Core now has a `FusedSharedExpertMLP` using `GroupedLinear(num_groups=1)` and a
fused SwiGLU path, but its implementation requires TE 2.14+, a supported block-scaled recipe, bias-free
linears and a GLU interleave size. That existing design is the reference for any shared-expert
experiment; do not invent an unrelated custom CUDA kernel first.

DeepSeek's current DeepGEMM is also not an Ada shortcut. Its upstream requirements list only SM90 and
SM100. Its Mega-MoE path additionally fuses expert-parallel communication for a multi-process symmetric
memory launch, which addresses a different regime from Aster's one-GPU laptop training workload.

## Decision

Do not attempt to install official FlashKDA or FlashMLA on this SM89 laptop and do not write an
Ada port before profiling the verified FLA/Transformer Engine paths. The next evidence gates are:

1. matched end-to-end training profiles for dense, grouped MoE, and latent MoE;
2. reference versus grouped experts at the confirmed batch-2 workload;
3. shared-expert and optimizer-fragmentation isolation using existing TE mechanisms;
4. KDA Triton kernel traces at Aster's exact head/sequence shapes;
5. separate prefill and recurrent-decode inference measurements;
6. only if KDA remains a dominant hotspot, estimate an SM89 FlashKDA port against the expected
   benefit and required correctness coverage.

## Primary sources

- Moonshot FlashKDA README and build constraints: https://github.com/MoonshotAI/FlashKDA
- Moonshot FlashKDA design note: https://github.com/MoonshotAI/FlashKDA/blob/master/docs/20260420-flashkda-v1-deep-dive.md
- FLA KDA integration: https://github.com/fla-org/flash-linear-attention
- DeepSeek FlashMLA hardware requirements: https://github.com/deepseek-ai/FlashMLA
- DeepSeek DeepGEMM requirements and grouped APIs: https://github.com/deepseek-ai/DeepGEMM
- NVIDIA Transformer Engine grouped-MoE tutorial: https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/te_mixtral/tutorial_accelerate_hf_mixtral_with_te.html
- NVIDIA Transformer Engine releases: https://github.com/NVIDIA/TransformerEngine/releases
- NVIDIA Megatron Core fused shared expert: https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/shared_experts.py
