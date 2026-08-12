# GPU-resident 8-bit Muon state probe — 2026-08-12

## Decision being tested

The K3 scale gate was constrained by persistent optimizer state, not by KDA's
context-dependent state. CPU and NVMe optimizer offload are excluded from the
pretraining architecture because their PCIe and storage traffic would compromise
wall-clock throughput. This probe instead keeps all training state on the GPU.

The implementation follows the blockwise signed-linear state quantization described
in *Effective Quantization of Muon Optimizer States* (arXiv:2509.23106): Muon
momentum is stored as INT8 with one FP32 abs-max scale per 2,048 values. It is
dequantized for the update and requantized immediately afterward. The AdamW side of
the hybrid uses TorchAO AdamW8bit rather than a local reimplementation.

## Matched laptop evidence

Candidate: `aster_k3_latentmoe_868m_a483m`, sequence 2,048, micro-batch 1,
accumulation 8, CUTLASS MoE, per-head Muon, BF16 master parameters/gradients.

| Persistent optimizer state | Peak allocated | Peak reserved | Measured throughput | GPU utilization |
|---|---:|---:|---:|---:|
| FP32 Muon + fused AdamW | 10.50 GiB | 10.99 GiB | 2.61k tok/s | 100% median |
| INT8 Muon + TorchAO AdamW8bit | 8.39 GiB | 9.42 GiB | 3.26–3.40k tok/s | 91–95% |

The low-bit path recovered about 2.11 GiB of peak allocated memory in the warmed
probe. Its first CUDA step was excluded from throughput comparison because TorchAO
compilation made that cold step unrepresentative. The three measured warmed steps
were 3.40k, 3.28k, and 3.26k tok/s. This remains a short diagnostic, not a
convergence result.

The 868M model's BF16 parameter storage is about 1.62 GiB. Transformer Engine FP8
compute did not change that persistent storage and measured 10.56 GiB peak under
the ordinary Muon recipe, slightly above BF16 compute because eligible GEMMs alone
were quantized while master parameters, gradients, optimizer state, and CUTLASS
expert weights remained BF16.

## Validation completed

- Odd-sized block/tail quantization error is bounded by half the block scale.
- Persistent Muon state contains INT8 momentum and FP32 block scales, with no FP32
  momentum buffer.
- Save/load followed by the next optimizer step is bit-identical on CPU.
- A real CUDA hybrid step completed with every optimizer state resident on GPU.
- The WSL CUDA runtime uses PyTorch 2.13.0+cu130; TorchAO is pinned to 0.17.0.

## Remaining promotion gates

1. Repeat from a clean source-pinned checkout at 8K context and record full phase
   timing, VRAM, utilization, and throughput.
2. Run matched-token learning curves against FP32-state per-head Muon. Low-bit state
   is promoted only if convergence and stability remain equivalent within the
   campaign's declared statistical tolerance.
3. Test interrupted full-checkpoint recovery on CUDA, including the TorchAO Adam
   partition.
4. Only after those gates decide whether 868M is the laptop-trainable final scale or
   whether an intermediate K3 scale is required.

No evidence in this probe supports replacing BF16 master training parameters with
FP8 on this Ada GPU. FP8 compute remains an optional backend to retest on cloud
hardware with native high-throughput FP8 support.

## Long-context correction

The first source-pinned 4K and 8K attempts failed during backward near the WSL/WDDM
memory ceiling because the scale candidate still used per-block checkpoint
boundaries (`checkpoint_segment_size: 1`). This is not KDA state growth: each layer
was retaining another full `[batch, context, d_model]` checkpoint input.

Using four-block checkpoint segments, with identical model equations and no host
offload, produced:

| Context | Peak allocated | Peak reserved | Measured throughput | GPU utilization |
|---|---:|---:|---:|---:|
| 4K | 5.88 GiB | 7.17 GiB | 4.29k tok/s | 85% sampled |
| 8K | 7.40 GiB | 8.68 GiB | 4.04–4.09k tok/s | 91–97% sampled |

The successful 8K run retains about 2.5 GiB of process-memory headroom on the 12
GiB laptop GPU. Segment size four is therefore the current 868M laptop execution
candidate; a clean source-pinned repeat is required after committing the model
config and campaign protocol.

That clean repeat on commit `48afd600824e0d792f96d1e91d55292448a43cab`
completed at 4,105.5 tok/s median, 97% median GPU utilization, 7.30 GiB peak
allocated, 7.64 GiB peak reserved, and 8,432 MiB median process memory. The
source-pinned evidence is under
`runs/architecture-campaign/k3-scale-frontier-int8-8k-seg4-48afd60`.

A bounded segment-size-two comparison completed at 4,042.6 tok/s median and 6.29
GiB peak allocated. Segment four is about 1.6% faster and remains the 8K default;
segment two is the lower-memory fallback for later context-extension stages.

## Matched 868M quality gate

The source-pinned 1,048,576-token comparison used identical named
initialization, data order, BF16 compute, K3 per-head Muon equations, and CUTLASS
experts. The only treatment was persistent optimizer-state representation.

| State | Final eval loss | Median tok/s | Mean GPU util. | Peak training VRAM |
|---|---:|---:|---:|---:|
| FP32 Muon | 6.678311 | 3,456.97 | 89.55% | 7.338 GiB |
| GPU INT8 Muon | 6.704460 | 3,160.49 | 77.66% | 4.991 GiB |

INT8 reduced total training VRAM by about 32%, but was 9.4% slower and ended
0.39% higher in eval loss at this short horizon. The 868M laptop default remains
FP32-state Muon when it fits. INT8 state is retained for a model/context tier
that otherwise cannot run fully on-device. Evidence is stored under
`runs/optimizer-campaign/muon-state-confirm-1m-1d95a47`.

## Larger laptop tier

The 1.448B-total / 568.2M-active candidate also fits fully on-device at 8K with
INT8 Muon state. Two-block recomputation measured 3.22k tok/s, 97% median GPU
utilization, 9.31 GiB peak allocation, and 10.38 GiB peak reservation. A clean
four-block run technically fit but nearly filled the device (11.95 GiB reserved)
and fell to 1.38k tok/s. Its default is therefore segment 2. No CPU or NVMe
offload is used in either model.

This is a fit and throughput result, not a quality promotion. The 1.45B tier must
beat the 868M tier on the source-pinned matched-data learning-curve gate before
becoming the pretraining model.
