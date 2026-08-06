import torch

from asterlm.cache import LatentLayerCache
from asterlm.quantization.kv import dequantize_tensor, quantize_tensor
from asterlm.quantization.loqt import LoQTLinear, merge_loqt_modules


def test_int4_and_hadamard_roundtrip_reduce_storage():
    torch.manual_seed(1)
    x = torch.randn(2, 7, 48, dtype=torch.bfloat16)
    for scheme, max_rmse in (("int4", 0.18), ("hadamard_int4", 0.18)):
        q = quantize_tensor(x, scheme, group_size=16)
        restored = dequantize_tensor(q, dtype=x.dtype)
        assert restored.shape == x.shape
        assert float((restored.float() - x.float()).square().mean().sqrt()) < max_rmse
        assert q.num_bytes < x.numel() * x.element_size()


def test_hot_cold_cache_quantizes_old_chunks():
    cache = LatentLayerCache(
        cache_dtype="hadamard_int4",
        group_size=16,
        recent_tokens=4,
        chunk_tokens=4,
        quantize_rope=False,
    )
    latent = torch.randn(1, 12, 16, dtype=torch.bfloat16)
    rope = torch.randn(1, 12, 8, dtype=torch.bfloat16)
    cache.append(latent, rope, window_size=None, sink_tokens=0)
    assert cache.length == 12
    assert cache.cold_length >= 4
    restored = cache.materialize_latent(dtype=torch.bfloat16, device=torch.device("cpu"))
    assert restored is not None and restored.shape == latent.shape
    assert cache.num_bytes < (latent.numel() + rope.numel()) * 2


def test_loqt_backward_and_merge():
    torch.manual_seed(2)
    layer = LoQTLinear(32, 48, rank=8, group_size=16)
    x = torch.randn(3, 32, requires_grad=True)
    loss = layer(x).square().mean()
    loss.backward()
    assert x.grad is not None
    assert layer.b.grad is not None
    before = layer.qweight.clone()
    with torch.no_grad():
        layer.b.add_(0.01)
    stats = merge_loqt_modules(layer)
    assert stats.modules == 1
    assert not torch.equal(before, layer.qweight)
    assert torch.count_nonzero(layer.b) == 0
