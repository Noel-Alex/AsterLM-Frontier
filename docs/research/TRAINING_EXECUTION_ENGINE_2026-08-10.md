# Aster training execution engine

Status: active engineering workstream. The current promoted adapter is the lean
single-process `aster_local` plan. External runtimes remain challengers until they
pass the same numerical, recovery, quality, and sustained-throughput gates.

The exact upstream checkouts used for adapter work are pinned in
`configs/research/upstream_sources_2026-08-10.yaml`. A locally cached checkout is a
research input, not evidence that its runtime is installed, compatible, or promoted.

## Why this is separate from model architecture

A good KDA, MLA, MoE, mHC, MTP, or optimizer design can look poor when executed as
many small Python-dispatched kernels, when routing causes device-to-host
synchronization, or when compilation repeatedly specializes on dynamic shapes. The
architecture campaign therefore records two independent axes:

1. mathematical/model merit: loss, downstream quality, long-context retrieval,
   stability, capacity and sample efficiency;
2. execution maturity: useful FLOPs, tokens/s, utilization, memory, launch overhead,
   graph breaks, host stalls, scaling and recovery.

Low utilization is an execution defect to diagnose. It is never, by itself, a reason
to reject an architecture.

## Upstream decision

SGLang is an inference engine and an RL rollout backend. It can matter enormously in
post-training, but it is not a pretraining runtime.

Megatron Core is the primary feature/reference source for NVIDIA scale-out plans. Its
current relevant components include MoE token-dispatch/permutation fusion, grouped
FP8 GEMM, shared-expert overlap, router fusion, MLA, Muon, fine-grained recomputation
and offload, CUDA-graph scopes, communication overlap, multiple parallelism axes and
distributed checkpointing.

TorchTitan is the primary PyTorch-native execution reference. Its relevant pieces
include `torch.compile`, selective activation checkpointing, Float8, FSDP2, tensor,
pipeline and context parallelism, asynchronous distributed checkpoints,
checkpointable data loading, structured logs, profiling, and fault-tolerance hooks.

DeepSpeed remains a measured candidate for ZeRO, CPU/NVMe offload, AutoTP, pipeline
parallelism and provider-specific configurations. On one GPU, offload can increase
model capacity but may reduce tokens/s. Capacity and time are measured separately;
neither is assumed from a framework name.

## Execution-plan API

Every run resolves a plan before training and stores it in `run_manifest.json`.
The plan records:

- exact device/process topology;
- engine and distributed strategy;
- precision and compile policy;
- static versus dynamic shape policy;
- activation/offload and graph-capture policy;
- human-readable decisions.

`execution_backend` accepts `auto`, `aster_local`, `megatron_core`, `torchtitan`, and
`deepspeed`. Only an actually validated adapter may execute. Requesting an
unimplemented adapter fails loudly instead of running a different engine while the
manifest claims otherwise.

## Target-specific plans

### Laptop, one Ada GPU

- maximize safe microbatch before accumulation;
- bucket sequence lengths so compiled graphs stay stable;
- keep the data pipeline ahead of the GPU with pinned, nonblocking transfers;
- fuse loss, norm, optimizer and routing work when parity permits;
- keep KDA/MLA and MoE kernel choices shape-aware;
- use BF16 or FP8 by measured operator coverage, never by label alone;
- evaluate CUDA graphs only after dynamic MoE routing and optimizer semantics are
  capture-safe;
- offload only when the extra capacity improves final quality enough to repay the
  throughput cost.

### Remote single GPU

- re-autotune for the exact SM, VRAM, memory bandwidth and software image;
- cache compiled extensions, datasets and tokenizer artifacts in provider volumes;
- use larger microbatches and context buckets where memory permits;
- stage checkpoints asynchronously and upload durable milestones to Hugging Face;
- never copy a laptop kernel schedule merely because its math is identical.

### Remote multi-GPU

- choose data, tensor, pipeline, context and expert parallelism from measured topology;
- overlap collectives only after numerical and restart parity;
- use distributed checkpoints that can be resharded and returned to the laptop;
- detect stragglers, rank failure and data-position drift;
- benchmark Megatron Core, TorchTitan and DeepSpeed adapters against an Aster-native
  PyTorch distributed control.

## Mandatory optimizer loop

The engine campaign is iterative rather than a one-time framework choice:

1. capture an operator trace and phase timings;
2. identify GPU bubbles, host synchronization, allocator pressure and graph breaks;
3. choose an upstream kernel/runtime or implement the smallest missing generic piece;
4. prove forward, loss, input-gradient and every-parameter-gradient parity;
5. test optimizer update, checkpoint save/resume and deterministic data position;
6. tune microbatch, accumulation, shape buckets, compile mode and kernel schedules;
7. soak under evaluation and checkpoint pressure;
8. record cold start, warm sustained throughput, utilization distribution and memory;
9. promote only when quality is unchanged and the end-to-end run wins.

Energy is recorded for research statistics only and never participates in promotion.

## Promotion gates

An adapter is not production-ready until all applicable gates pass:

- exact architecture/config identity and checkpoint compatibility;
- bounded BF16/FP8 forward, loss, input-gradient and per-parameter-gradient error;
- optimizer-step trajectory parity over a short real-data run;
- no dropped MoE tokens and equivalent router/load-balancing semantics;
- KDA/MLA cache and recurrent-state equivalence;
- save, resume, preemption and cross-engine checkpoint recovery;
- checkpointable data position with no silent replay or skip;
- W&B/JSONL/Hugging Face artifact continuity;
- cold-start and warmed measurements reported separately;
- sustained tokens/s, median/p10/p90 utilization, phase time, graph breaks, CPU wait,
  allocated/reserved peak memory, and fragmentation;
- end-to-end win including evaluation and checkpoint intervals;
- no statistically meaningful quality regression at equal data and tokens.

## Near-term implementation order

1. Finish the KDA context crossover and MoE shape/work-granularity sweep.
2. Add reusable static sequence buckets and a runtime autotuning ledger.
3. Fuse/compile the loss and optimizer regions independently of dynamic model regions.
4. Prototype CUDA-graph capture for dense fixed-shape controls, then MoE with routing
   replay/static buffers.
5. Build a Megatron Core compatibility spike for grouped FP8 MoE and remote
   distributed execution.
6. Build a TorchTitan adapter/control for PyTorch-native FSDP2/TP/CP and distributed
   checkpointing.
7. Benchmark DeepSpeed offload locally and ZeRO/AutoTP remotely where memory or
   topology makes them relevant.
8. Integrate SGLang only in the post-training rollout plane and later serving plane.

No external runtime is adopted wholesale. Components are reused when their license,
hardware support, checkpoint semantics, measured performance and quality match the
Aster campaign.
