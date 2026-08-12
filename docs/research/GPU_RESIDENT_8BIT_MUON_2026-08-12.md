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
