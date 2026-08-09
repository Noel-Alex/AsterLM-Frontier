<div align="center">

# 🐳 Native Sparse Attention

[![arxiv](https://img.shields.io/badge/arXiv-2502.11089-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2502.11089)

</div>

Efficient Triton implementations for [Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](https://arxiv.org/abs/2502.11089).

<div align="center">
  <img width="400" alt="image" src="https://github.com/user-attachments/assets/ace2920d-3894-4556-8039-b70861742551">
</div>

## News

- [2025-03-15] We now add full support for NSA training with `flame`.
- [2025-02-25] Introduce an online top‑k selection kernel that avoids materializing the attention matrix during selection.
- [2025-02-24] Add support for a fused Triton kernel combining selected attention with sliding attention.
- [2025-02-21] Enable handling of a variable number of selected blocks for queries across different positions and batches.

### Setup

To get started, clone the `native-sparse-attention` repository and install the required dependencies:

```bash
git clone https://github.com/fla-org/native-sparse-attention.git
cd native-sparse-attention
git submodule update --init --recursive
pip install .
```

## Usage

To test the correctness of NSA:
```py
pytest tests/test_nsa.py
```

To validate the correctness of NSA with top‑k selection, run the command below. Please note that the initial trial may take some time as the kernel compiles, but subsequent runs will be faster.
```py
pytest tests/test_nsa_with_compression.py
```

To measure the efficiency of NSA:
```py
python benchmarks/benchmark_nsa.py
```

To direct use our NSA kernel:
```py
from native_sparse_attention.ops.parallel import parallel_nsa

B, T, H, HQ, D = 4, 2048, 4, 64, 64
block_size = 64
window_size = 64

q = torch.randn((B, T, HQ, D), dtype=dtype, device='cuda').requires_grad_(True)
k = torch.randn((B, T, H, D), dtype=dtype, device='cuda').requires_grad_(True)
v = torch.randn((B, T, H, D), dtype=dtype, device='cuda').requires_grad_(True)
g_slc = torch.rand((B, T, HQ), dtype=dtype, device='cuda').requires_grad_(True)
g_swa = torch.rand((B, T, HQ), dtype=dtype, device='cuda').requires_grad_(True)

# randomly generated block indices
block_indices = torch.full((B, T, H, S), T, dtype=torch.long, device=device)
for b in range(B):
    for t in range(T):
        for h in range(H):
            i_i = torch.randperm(max(1, triton.cdiv(t, block_size)))[:S]
            block_indices[b, t, h, :len(i_i)] = i_i
block_indices = block_indices.sort(-1)[0]
block_counts = torch.randint(1, S + 1, (B, T, H), device=device)

parallel_nsa(
    q=q,
    k=k,
    v=v,
    g_slc=g_slc,
    g_swa=g_swa,
    block_indices=block_indices,
    block_counts=block_counts,
    block_size=block_size,
    window_size=window_size,
)

# variable-length inputs are supported as well
# randomly split the sequence into N segments
N, T = 4, 2048
offsets = torch.cat([
    torch.tensor([0], dtype=torch.long),
    torch.arange(16, T)[torch.randperm(T - 1)[:N-1]],
    torch.tensor([T], dtype=torch.long)
], 0).cuda().sort()[0]
# seq-first required for inputs with variable lengths
q = torch.rand((1, T, HQ, D), dtype=dtype, device='cuda').requires_grad_(True)
k = torch.rand((1, T, H, D), dtype=dtype, device='cuda').requires_grad_(True)
v = torch.rand((1, T, H, D), dtype=dtype, device='cuda').requires_grad_(True)
g_slc = torch.rand((B, T, HQ), dtype=dtype, device='cuda').requires_grad_(True)
g_swa = torch.rand((B, T, HQ), dtype=dtype, device='cuda').requires_grad_(True)

# randomly generated block indices
block_indices = torch.full((B, T, H, S), T, dtype=torch.long, device=device)
for b in range(B):
    for t in range(T):
        for h in range(H):
            i_i = torch.randperm(max(1, triton.cdiv(t, block_size)))[:S]
            block_indices[b, t, h, :len(i_i)] = i_i
block_indices = block_indices.sort(-1)[0]
block_counts = torch.randint(1, S + 1, (B, T, H), device=device)

parallel_nsa(
    q=q,
    k=k,
    v=v,
    g_slc=g_slc,
    g_swa=g_swa,
    block_indices=block_indices,
    block_counts=block_counts,
    block_size=block_size,
    window_size=window_size,
    cu_seqlens=offsets
)
```

## Benchmarks

```sh
Performance:
         T        nsa     nsa_bwd      flash   flash_bwd
0    128.0   0.091168    0.672992   0.020128    0.161504
1    256.0   0.189408    1.222848   0.045024    0.225056
2    512.0   0.435616    2.363264   0.105664    0.503264
3   1024.0   1.043200    5.091552   0.296944    1.323456
4   2048.0   2.322016   11.124559   0.970208    4.076928
5   4096.0   4.869712   23.082577   3.520352   14.193248
6   8192.0   9.953824   49.575199  13.464992   52.566914
7  16384.0  20.164879  116.297920  53.633568  204.353607
```
<div align="center">
<img width="400" alt="image" src="https://github.com/user-attachments/assets/efc25313-b058-47ae-b96e-ed67c62c134d">
</div>

## Citations

```bibtex
@inproceedings{Yuan2025NativeSA,
    title   = {Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention},
    author  = {Jingyang Yuan and Huazuo Gao and Damai Dai and Junyu Luo and Liang Zhao and Zhengyan Zhang and Zhenda Xie and Y. X. Wei and Lean Wang and Zhiping Xiao and Yuqing Wang and Chong Ruan and Ming Zhang and Wenfeng Liang and Wangding Zeng},
    year    = {2025},
    url     = {https://api.semanticscholar.org/CorpusID:276408911}
}
```
