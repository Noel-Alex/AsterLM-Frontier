# Ada KDA and MoE utilization repair — 2026-08-10

Status: systems evidence only. Synthetic-token throughput cannot promote or reject
an architecture and makes no model-quality claim. Raw JSON/operator traces live in
ignored run storage under `runs/architecture-campaign/` and
`runs/frontier-vnext2/cutlass-moe-screen-20260810/`.

## KDA execution repair

The installed PyTorch CUDA 13.0 runtime had been combined with NVCC/CCCL 13.3 and
CUDA 13.0 headers. That mixture prevented FLA's TileLang KDA backward kernel from
compiling. An isolated, version-matched CUDA 13.0.88 compiler prefix made the
upstream TileLang kernel usable without changing PyTorch's runtime packages.

At sequence 2,048, batch 1, accumulation 1, BF16 and AdamW:

| KDA execution | tokens/s | median GPU util. | forward | backward |
|---|---:|---:|---:|---:|
| FLA baseline | 5,004.82 | 78% | 92.07 ms | 242.60 ms |
| FLA + TileLang backward | 5,872.61 | 83% | 99.89 ms | 213.13 ms |
| FLA + TileLang + `torch.compile` surroundings | 6,746.39 | 87% | 89.79 ms | 204.60 ms |
| compiled dense MLA control | 7,803.12 | 98% | 55.35 ms | 184.86 ms |

TileLang improved KDA by 17.3%; compilation raised the total gain to 34.8% over the
original KDA path. KDA still trails the compiled MLA control by 13.5% at 2K. This is
short-context optimization debt, not a KDA rejection. The mandatory follow-up is a
4K/8K/16K training and decode sweep plus retrieval evaluation through 128K.

## MoE root cause

The Transformer Engine grouped route accepts dynamic expert split sizes but converts
them to a Python list. Under `torch.compile`, changing split values caused repeated
specialization and graph-limit failures. The operator trace also showed 643 scalar
device-to-host copies and about 28 ms CPU time in `_local_scalar_dense`, plus costly
`index_add_`, sorting and padding. Compilation made this path slower and is rejected
for the current TE implementation; MoE itself remains eligible.

The Apache-2.0 MegaBlocks `nv_grouped_gemm` backend was built at upstream commit
`efe8c40eaf4c8ef57191e0ea9aa4117aa5b1a8f2` for SM89/CUDA 13.0. It supplies
autograd-enabled CUTLASS grouped GEMMs and fused top-k permute/unpermute. Aster's
adapter keeps the existing per-expert checkpoint Parameters, performs no token
dropping, and passed CUDA output/input-gradient/all-parameter-gradient parity against
the reference route.

The first adapter stacked all expert weights for every microbatch. The trace measured
about 1.09 GB of temporary `cat` allocation in one profiled step. The repaired adapter
caches detached packed weights across gradient-accumulation microbatches and uses a
custom autograd bridge to return weight gradients to the canonical Parameters. Cache
versions force a refresh after every optimizer update or checkpoint load.

## Controlled MoE result

The fair batch-1 comparison used the same 24-layer model, 3:1 KDA/MLA pattern,
top-2/E8 dropless routing, shared expert, TE FP8 surroundings, APOLLO Mini, sequence
2,048 and accumulation 16. Only routed-expert execution changed.

| Routed implementation | tokens/s | median GPU util. | peak VRAM | forward | backward |
|---|---:|---:|---:|---:|---:|
| TE grouped + TileLang KDA | 5,826.18 | 41% | 5.212 GiB | 2,869.8 ms | 2,616.2 ms |
| CUTLASS, uncached weights | 6,757.14 | 53% | 5.407 GiB | 2,447.7 ms | 2,200.4 ms |
| CUTLASS, cached weights | 7,818.49 | 54% | 5.414 GiB | 2,166.2 ms | 1,884.8 ms |

The cached CUTLASS route is 34.2% faster than the fair TE control. It cuts forward
time 24.5% and backward time 28.0%. Larger microbatches expose enough work to occupy
the GPU more effectively:

| Mode | microbatch × accumulation | tokens/s | median / p90 GPU util. | peak VRAM | status |
|---|---:|---:|---:|---:|---|
| safe laptop profile | 2 × 8 | 12,334.39 | 82% / 92% | 8.265 GiB | fits with headroom |
| max-throughput probe | 3 × 6 | 14,238.36 | 89% / 96% | 10.917 GiB | provisional |

The batch-3 process had three extremely slow cold warmups near the memory ceiling and
only about 0.33 GiB allocator headroom under the project's 11.25 GiB gate. Its steady
steps were fast, but it is not the safe default until repeated cold starts, checkpoint
save/resume, evaluation-at-peak and a long soak pass without OOM or throughput decay.

## Remaining work

- Remove or amortize the host expert-count launch dependency; inspect CUDA-graph and
  device-side masked grouped-GEMM designs.
- Tune the SM89 CUTLASS grouped GEMM tile/schedule for Aster's E4/E8, 768×704 shapes.
- Build and quality-gate a true FP8 expert grouped-GEMM branch. The current CUTLASS
  grouped expert GEMMs are BF16 even though surrounding TE layers use FP8.
- Re-run dense and MoE controls under identical batch/token/wall-clock conditions.
- Run the KDA context crossover and retrieval campaign before architecture promotion.

Power and energy were recorded in every profile for research statistics only. They
did not rank, reject or select any path.
