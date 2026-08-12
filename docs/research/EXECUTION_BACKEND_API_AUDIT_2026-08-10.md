# Source-pinned execution backend API audit

Date: 2026-08-10

This audit uses only the exact upstream commits locked in
`configs/research/upstream_sources_2026-08-10.yaml`. It identifies realistic Aster
integration boundaries; it does not promote any external runtime.

## Current result

The immediate execution strategy remains:

1. keep `aster_local` as the laptop and numerical-control engine;
2. build DeepSpeed as the first arbitrary-`nn.Module` wrapper candidate;
3. build TorchTitan as the primary PyTorch-native distributed control;
4. treat Megatron Core as the highest-performance NVIDIA scale-out target and a
   source of reusable MoE/FP8/optimizer/checkpoint components, with a deeper model
   port rather than pretending it is a drop-in wrapper.

This order is about integration risk, not an assumed performance ranking. Every
candidate must beat the same end-to-end gate on the actual laptop or remote GPU.

## Megatron Core

Pinned commit: `24bad8e677d22625d86ef2a54c9506b6e4992c93`.

Relevant concrete APIs in this checkout include:

- `megatron/core/distributed/distributed_data_parallel.py:22`,
  `DistributedDataParallel`, with contiguous gradient buffers, optional gradient
  reduce overlap, reduce-scatter and separate accumulation dtype;
- `megatron/core/process_groups_config.py:27`, `ProcessGroupCollection`, and
  `megatron/core/parallel_state.py:601`, `initialize_model_parallel`;
- `megatron/core/optimizer/optimizer_config.py:139`, `OptimizerConfig`, including
  Adam/SGD/Muon selection, BF16/FP16, FP8-aware distributed-optimizer controls and
  precision-aware optimizer-state dtypes;
- `megatron/core/optimizer/__init__.py:991`, `get_megatron_optimizer`;
- `megatron/core/transformer/moe/experts.py:166`, grouped-expert submodule
  contracts and `TEGroupedMLP` immediately below them;
- `megatron/core/transformer/moe/token_dispatcher.py:1748`,
  `MoEFlexTokenDispatcher`, with DeepEP, HybridEP and NCCL-EP backends;
- `megatron/core/transformer/multi_latent_attention.py:130`,
  `MultiLatentAttention`;
- `megatron/core/dist_checkpointing/serialization.py:341` and `:69`, sharded
  `save`/`load`, plus
  `dist_checkpointing/strategies/fully_parallel.py:46` for fully parallel save.

The integration is not a call around an arbitrary Aster module. MCore DDP and
optimizer paths expect MCore transformer/process-group configuration, parameter
metadata, distributed state and compatible sharded state dictionaries. Its expert
and attention modules also use MCore `ModuleSpec`, Transformer Engine and process
group conventions. A superficial wrapper would either bypass the useful features or
silently change Aster math/checkpoint semantics.

The planned spike therefore has two layers:

- component parity: compare MCore/Transformer Engine grouped expert operations,
  routing permutations and optimizer kernels behind the existing Aster model API;
- native remote adapter: map a frozen Aster architecture identity into MCore model
  specs, process groups and sharded state, then prove tensor-by-tensor conversion and
  restart parity.

On a single laptop GPU, MCore distributed communication is not itself a speedup.
The locally relevant candidates are grouped expert kernels, FP8 coverage, fused
optimizer work, recomputation/offload policy and CUDA-graph regions. Remote
multi-GPU execution is where its TP/PP/CP/EP and communication overlap become
first-class candidates.

## TorchTitan

Pinned commit: `5c0b804cecd54313d7cfea41a386b3da1a1d19dc`.

The usable model boundary is
`torchtitan/protocols/model_spec.py:33`, `ModelSpec`. It carries the model class,
configuration, parallelization function, pipeline function, optimizer/loss
factories and state-dict adapters. The main trainer is
`torchtitan/trainer.py:59`; non-pipeline construction applies
`model_spec.parallelize_fn` around `trainer.py:445`. Current in-tree registrations
for DeepSeek V3 and Kimi K2.7 demonstrate that non-Llama architectures are expected,
not accidental extensions.

Checkpointing is centered on
`torchtitan/components/checkpoint.py:176`, `CheckpointManager`, with explicit
`save` at `:706`, `load` at `:801`, and `close` at `:539`. The manager composes model,
optimizer, scheduler, dataloader and training progress around PyTorch distributed
checkpoint state.

TorchTitan is therefore a clean custom-model target, but it owns more of the training
application than a thin engine wrapper. The correct adapter is an Aster `ModelSpec`
plus state-dict/data-position adapters, not calls to TorchTitan from inside every
Aster step. The first control should keep Aster math and loss unchanged while using
TorchTitan parallelization/checkpoint orchestration. FSDP2/TP/CP/PP variants follow
only after single-rank numerical identity.

## DeepSpeed

Pinned commit: `cf44300453eb0af79ed84ed8f1cb49d57478bd76`.

`deepspeed/__init__.py:93`, `initialize`, accepts an arbitrary `torch.nn.Module`, an
existing optimizer, model parameters, scheduler and a configuration dictionary. It
returns `DeepSpeedEngine` (`deepspeed/runtime/engine.py:249`) and wrapped optimizer,
dataloader and scheduler objects. The engine exposes `backward` at `:3160`, `step`
at `:3360`, `load_checkpoint` at `:4214` and `save_checkpoint` at `:4692`.
`deepspeed/profiling/flops_profiler/profiler.py:30` can profile a standalone module
or engine. `PipelineModule` is at `runtime/pipe/module.py:86`, whose source explicitly
states that pipeline parallelism is incompatible with ZeRO-2 and ZeRO-3.

This is the lowest-friction first wrapper candidate, but its engine assumes control
over distributed initialization, gradient accumulation boundaries, scaling,
clipping, scheduler stepping and checkpoint layout. The adapter must therefore:

- expose Aster's exact accumulation and token-count semantics;
- prevent both Aster and DeepSpeed from stepping/clipping/scaling the same update;
- include Aster RNG, dataloader cursor, architecture identity and experiment metadata
  in recoverable client state;
- validate ordinary, ZeRO and offload checkpoint restoration and conversion back to
  the portable Aster/Hugging Face format;
- measure compilation interactions and never combine feature flags merely because
  both systems accept them.

## Adapter contract and promotion sequence

All candidates implement the same conceptual boundary:

- prepare an unchanged architecture or an explicitly proven translation;
- run forward/backward/optimizer update with phase timings;
- expose unwrapped named parameters and stable state-dict keys;
- save and restore model, optimizer, scheduler, scaler, RNG and exact data position;
- report runtime/package/source identity and all effective execution decisions;
- surface unsupported combinations before allocation or distributed launch.

The first milestone for each adapter is a CPU/tiny-tensor contract test. The second
is exact FP32 single-rank forward/loss/gradient/update parity. BF16 and architecture
features are enabled one at a time. Checkpoint interruption and cross-engine restore
come before performance claims. Only then are sustained laptop or remote profiles,
MoE routing stress, KDA state parity, checkpoint/evaluation overlap and quality
campaigns allowed to decide promotion.

No framework name is treated as evidence of speed. Cold start, warm tokens/s,
utilization distribution, phase timings, memory, graph breaks and end-to-end quality
remain the decision evidence; energy remains a recorded statistic only.
