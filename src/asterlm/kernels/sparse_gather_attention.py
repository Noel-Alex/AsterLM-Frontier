from __future__ import annotations

import torch

try:  # Triton is unavailable in native Windows PyTorch environments.
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - platform/package dependent
    triton = None
    tl = None


def sparse_gather_attention_reference(
    query: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    attention_sink: torch.Tensor,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """Differentiable PyTorch oracle for indexed shared-KV attention."""

    if query.ndim != 4 or kv.ndim != 3 or indices.ndim != 3:
        raise ValueError("expected query [B,S,H,D], kv [B,N,D], indices [B,S,K]")
    batch, sequence, heads, head_dim = query.shape
    if kv.shape[0] != batch or kv.shape[2] != head_dim:
        raise ValueError("KV geometry does not match query")
    if indices.shape[:2] != (batch, sequence):
        raise ValueError("index batch/sequence geometry does not match query")
    if attention_sink.shape != (heads,):
        raise ValueError("attention_sink must contain one scalar per query head")
    if indices.shape[-1] == 0:
        return torch.zeros_like(query)

    valid = (indices >= 0) & (indices < kv.shape[1])
    safe_indices = indices.clamp(0, max(kv.shape[1] - 1, 0)).long()
    gather_source = kv.unsqueeze(1).expand(-1, sequence, -1, -1)
    selected = torch.gather(
        gather_source,
        2,
        safe_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim),
    )
    attention_scale = head_dim**-0.5 if scale is None else scale
    scores = torch.einsum(
        "bshd,bskd->bshk", query.float(), selected.float()
    ) * attention_scale
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
    sink_scores = attention_sink.float().view(1, 1, heads, 1).expand(
        batch, sequence, -1, -1
    )
    probabilities = torch.cat((scores, sink_scores), dim=-1).softmax(dim=-1)[..., :-1]
    output = torch.einsum("bshk,bskd->bshd", probabilities, selected.float())
    return output.to(query.dtype)


if triton is not None:

    @triton.jit
    def _sparse_gather_attention_forward(
        query,
        kv,
        indices,
        attention_sink,
        output,
        logsumexp,
        sequence: tl.constexpr,
        heads: tl.constexpr,
        head_dim: tl.constexpr,
        selected_count: tl.constexpr,
        kv_count: tl.constexpr,
        scale: tl.constexpr,
        block_d: tl.constexpr,
        block_k: tl.constexpr,
    ):
        program = tl.program_id(0)
        head = program % heads
        token_linear = program // heads
        token = token_linear % sequence
        batch = token_linear // sequence

        d_offsets = tl.arange(0, block_d)
        d_valid = d_offsets < head_dim
        query_offsets = ((batch * sequence + token) * heads + head) * head_dim + d_offsets
        q = tl.load(query + query_offsets, mask=d_valid, other=0.0).to(tl.float32)

        maximum = -float("inf")
        denominator = 0.0
        accumulator = tl.zeros((block_d,), dtype=tl.float32)
        index_base = (batch * sequence + token) * selected_count

        for start in range(0, selected_count, block_k):
            k_offsets = start + tl.arange(0, block_k)
            within_k = k_offsets < selected_count
            selected = tl.load(indices + index_base + k_offsets, mask=within_k, other=-1)
            selected_valid = within_k & (selected >= 0) & (selected < kv_count)
            kv_offsets = (batch * kv_count + selected[:, None]) * head_dim + d_offsets[None, :]
            values = tl.load(
                kv + kv_offsets,
                mask=selected_valid[:, None] & d_valid[None, :],
                other=0.0,
            ).to(tl.float32)
            scores = tl.sum(values * q[None, :], axis=1) * scale
            scores = tl.where(selected_valid, scores, -float("inf"))

            block_maximum = tl.max(scores, axis=0)
            new_maximum = tl.maximum(maximum, block_maximum)
            had_values = maximum != -float("inf")
            has_values = new_maximum != -float("inf")
            old_scale = tl.where(
                had_values & has_values, tl.exp(maximum - new_maximum), 1.0
            )
            probabilities = tl.where(
                selected_valid, tl.exp(scores - new_maximum), 0.0
            )
            accumulator = accumulator * old_scale + tl.sum(
                probabilities[:, None] * values, axis=0
            )
            denominator = denominator * old_scale + tl.sum(probabilities, axis=0)
            maximum = new_maximum

        sink = tl.load(attention_sink + head).to(tl.float32)
        new_maximum = tl.maximum(maximum, sink)
        old_scale = tl.where(
            maximum != -float("inf"), tl.exp(maximum - new_maximum), 0.0
        )
        sink_probability = tl.exp(sink - new_maximum)
        accumulator *= old_scale
        denominator = denominator * old_scale + sink_probability
        result = accumulator / denominator

        tl.store(output + query_offsets, result, mask=d_valid)
        tl.store(logsumexp + (batch * sequence + token) * heads + head, new_maximum + tl.log(denominator))

    @triton.jit
    def _sparse_gather_attention_backward(
        query,
        kv,
        indices,
        attention_sink,
        output,
        logsumexp,
        output_gradient,
        query_gradient,
        kv_gradient,
        sink_gradient,
        sequence: tl.constexpr,
        heads: tl.constexpr,
        head_dim: tl.constexpr,
        selected_count: tl.constexpr,
        kv_count: tl.constexpr,
        scale: tl.constexpr,
        block_d: tl.constexpr,
        block_k: tl.constexpr,
    ):
        program = tl.program_id(0)
        head = program % heads
        token_linear = program // heads
        token = token_linear % sequence
        batch = token_linear // sequence

        d_offsets = tl.arange(0, block_d)
        d_valid = d_offsets < head_dim
        query_offsets = ((batch * sequence + token) * heads + head) * head_dim + d_offsets
        q = tl.load(query + query_offsets, mask=d_valid, other=0.0).to(tl.float32)
        out = tl.load(output + query_offsets, mask=d_valid, other=0.0).to(tl.float32)
        dout = tl.load(output_gradient + query_offsets, mask=d_valid, other=0.0).to(tl.float32)
        lse = tl.load(logsumexp + (batch * sequence + token) * heads + head)
        delta = tl.sum(out * dout, axis=0)
        dq = tl.zeros((block_d,), dtype=tl.float32)
        index_base = (batch * sequence + token) * selected_count

        for start in range(0, selected_count, block_k):
            k_offsets = start + tl.arange(0, block_k)
            within_k = k_offsets < selected_count
            selected = tl.load(indices + index_base + k_offsets, mask=within_k, other=-1)
            selected_valid = within_k & (selected >= 0) & (selected < kv_count)
            kv_offsets = (batch * kv_count + selected[:, None]) * head_dim + d_offsets[None, :]
            values = tl.load(
                kv + kv_offsets,
                mask=selected_valid[:, None] & d_valid[None, :],
                other=0.0,
            ).to(tl.float32)
            scores = tl.sum(values * q[None, :], axis=1) * scale
            probabilities = tl.where(
                selected_valid, tl.exp(scores - lse), 0.0
            )
            value_dot_gradient = tl.sum(values * dout[None, :], axis=1)
            score_gradient = probabilities * (value_dot_gradient - delta) * scale
            dq += tl.sum(score_gradient[:, None] * values, axis=0)
            dkv = probabilities[:, None] * dout[None, :] + score_gradient[:, None] * q[None, :]
            tl.atomic_add(
                kv_gradient + kv_offsets,
                dkv,
                mask=selected_valid[:, None] & d_valid[None, :],
            )

        sink = tl.load(attention_sink + head).to(tl.float32)
        sink_probability = tl.exp(sink - lse)
        tl.atomic_add(sink_gradient + head, -sink_probability * delta)
        tl.store(query_gradient + query_offsets, dq, mask=d_valid)


class _TritonSparseGatherAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        kv: torch.Tensor,
        indices: torch.Tensor,
        attention_sink: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        if triton is None or not query.is_cuda:
            raise RuntimeError("Triton sparse attention requires CUDA and the triton package")
        if query.shape[-1] > 512:
            raise ValueError("Triton sparse attention currently supports head_dim <= 512")
        query = query.contiguous()
        kv = kv.contiguous()
        indices = indices.to(device=query.device, dtype=torch.int32).contiguous()
        attention_sink = attention_sink.contiguous()
        batch, sequence, heads, head_dim = query.shape
        selected_count = indices.shape[-1]
        output = torch.empty_like(query)
        logsumexp = torch.empty(batch, sequence, heads, device=query.device, dtype=torch.float32)
        block_d = triton.next_power_of_2(head_dim)
        _sparse_gather_attention_forward[(batch * sequence * heads,)](
            query,
            kv,
            indices,
            attention_sink,
            output,
            logsumexp,
            sequence=sequence,
            heads=heads,
            head_dim=head_dim,
            selected_count=selected_count,
            kv_count=kv.shape[1],
            scale=scale,
            block_d=block_d,
            block_k=32,
            num_warps=4,
        )
        ctx.save_for_backward(query, kv, indices, attention_sink, output, logsumexp)
        ctx.scale = scale
        return output

    @staticmethod
    def backward(ctx, output_gradient: torch.Tensor):
        query, kv, indices, attention_sink, output, logsumexp = ctx.saved_tensors
        batch, sequence, heads, head_dim = query.shape
        query_gradient = torch.empty_like(query, dtype=torch.float32)
        kv_gradient = torch.zeros_like(kv, dtype=torch.float32)
        sink_gradient = torch.zeros_like(attention_sink, dtype=torch.float32)
        block_d = triton.next_power_of_2(head_dim)
        _sparse_gather_attention_backward[(batch * sequence * heads,)](
            query,
            kv,
            indices,
            attention_sink,
            output,
            logsumexp,
            output_gradient.contiguous(),
            query_gradient,
            kv_gradient,
            sink_gradient,
            sequence=sequence,
            heads=heads,
            head_dim=head_dim,
            selected_count=indices.shape[-1],
            kv_count=kv.shape[1],
            scale=ctx.scale,
            block_d=block_d,
            block_k=32,
            num_warps=4,
        )
        return (
            query_gradient.to(query.dtype),
            kv_gradient.to(kv.dtype),
            None,
            sink_gradient.to(attention_sink.dtype),
            None,
        )


def sparse_gather_attention(
    query: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    attention_sink: torch.Tensor,
    *,
    scale: float | None = None,
    backend: str = "auto",
) -> torch.Tensor:
    """Dispatch indexed shared-KV attention without changing its mathematics."""

    if backend not in {"auto", "torch", "triton"}:
        raise ValueError("backend must be auto, torch, or triton")
    attention_scale = query.shape[-1] ** -0.5 if scale is None else scale
    use_triton = backend == "triton" or (
        backend == "auto" and triton is not None and query.is_cuda
    )
    if use_triton:
        if triton is None or not query.is_cuda:
            raise RuntimeError("Triton sparse attention requires CUDA and the triton package")
        if indices.shape[-1] == 0:
            return torch.zeros_like(query)
        return _TritonSparseGatherAttention.apply(
            query, kv, indices, attention_sink, float(attention_scale)
        )
    return sparse_gather_attention_reference(
        query,
        kv,
        indices,
        attention_sink,
        scale=float(attention_scale),
    )
