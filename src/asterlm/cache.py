from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import torch

from .quantization.kv import QuantizedTensor, dequantize_tensor, quantize_tensor

if TYPE_CHECKING:
    from .config import AsterConfig


CachePiece = torch.Tensor | QuantizedTensor


def _piece_length(piece: CachePiece) -> int:
    return int(piece.original_shape[1] if isinstance(piece, QuantizedTensor) else piece.shape[1])


def _piece_bytes(piece: CachePiece) -> int:
    if isinstance(piece, QuantizedTensor):
        return piece.num_bytes
    return int(piece.numel() * piece.element_size())


def _nested_tensor_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_nested_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_nested_tensor_bytes(item) for item in value)
    if hasattr(value, "states"):
        return _nested_tensor_bytes(value.states)
    return 0


def _materialize(pieces: list[CachePiece], dtype: torch.dtype, device: torch.device) -> torch.Tensor | None:
    if not pieces:
        return None
    tensors = [
        dequantize_tensor(piece, dtype=dtype).to(device)
        if isinstance(piece, QuantizedTensor)
        else piece.to(device=device, dtype=dtype)
        for piece in pieces
    ]
    return torch.cat(tensors, dim=1)


@dataclass
class LatentLayerCache:
    """Hot/cold latent cache for one global-attention layer.

    The newest tokens remain in their compute dtype. Older chunks can be stored as
    FP8/INT8/INT4, including a portable Hadamard-rotated INT4 research path. KDA layers
    are not represented here; they keep fixed-size recurrent state in ``AsterCache``.
    """

    cache_dtype: str = "bfloat16"
    group_size: int = 64
    recent_tokens: int = 512
    chunk_tokens: int = 1024
    quantize_rope: bool = False
    latent: torch.Tensor | None = None  # hot [B,S,L]
    rope_key: torch.Tensor | None = None  # hot [B,S,R]
    cold_latent: list[CachePiece] = field(default_factory=list)
    cold_rope: list[CachePiece] = field(default_factory=list)

    @property
    def cold_length(self) -> int:
        return sum(_piece_length(piece) for piece in self.cold_latent)

    @property
    def length(self) -> int:
        hot = 0 if self.latent is None else int(self.latent.shape[1])
        return self.cold_length + hot

    @property
    def num_bytes(self) -> int:
        total = sum(_piece_bytes(piece) for piece in self.cold_latent)
        total += sum(_piece_bytes(piece) for piece in self.cold_rope)
        if self.latent is not None:
            total += self.latent.numel() * self.latent.element_size()
        if self.rope_key is not None:
            total += self.rope_key.numel() * self.rope_key.element_size()
        return int(total)

    def materialize_latent(
        self, *, dtype: torch.dtype | None = None, device: torch.device | None = None
    ) -> torch.Tensor | None:
        if self.latent is not None:
            dtype = self.latent.dtype if dtype is None else dtype
            device = self.latent.device if device is None else device
        elif self.cold_latent:
            first = self.cold_latent[0]
            if isinstance(first, QuantizedTensor):
                dtype = first.source_dtype if dtype is None else dtype
                device = first.data.device if device is None else device
            else:
                dtype = first.dtype if dtype is None else dtype
                device = first.device if device is None else device
        else:
            return None
        assert dtype is not None and device is not None
        cold = _materialize(self.cold_latent, dtype, device)
        hot = None if self.latent is None else self.latent.to(device=device, dtype=dtype)
        if cold is None:
            return hot
        if hot is None:
            return cold
        return torch.cat((cold, hot), dim=1)

    def materialize_rope(
        self, *, dtype: torch.dtype | None = None, device: torch.device | None = None
    ) -> torch.Tensor | None:
        if self.rope_key is not None:
            dtype = self.rope_key.dtype if dtype is None else dtype
            device = self.rope_key.device if device is None else device
        elif self.cold_rope:
            first = self.cold_rope[0]
            if isinstance(first, QuantizedTensor):
                dtype = first.source_dtype if dtype is None else dtype
                device = first.data.device if device is None else device
            else:
                dtype = first.dtype if dtype is None else dtype
                device = first.device if device is None else device
        else:
            return None
        assert dtype is not None and device is not None
        cold = _materialize(self.cold_rope, dtype, device)
        hot = None if self.rope_key is None else self.rope_key.to(device=device, dtype=dtype)
        if cold is None:
            return hot
        if hot is None:
            return cold
        return torch.cat((cold, hot), dim=1)

    def iter_chunks(self, *, dtype: torch.dtype, device: torch.device):
        """Yield aligned latent/RoPE chunks without materializing the whole cache."""
        if len(self.cold_latent) != len(self.cold_rope):
            raise RuntimeError("latent and RoPE cold-cache chunk counts diverged")
        for latent_piece, rope_piece in zip(self.cold_latent, self.cold_rope, strict=True):
            latent = (
                dequantize_tensor(latent_piece, dtype=dtype).to(device)
                if isinstance(latent_piece, QuantizedTensor)
                else latent_piece.to(device=device, dtype=dtype)
            )
            rope = (
                dequantize_tensor(rope_piece, dtype=dtype).to(device)
                if isinstance(rope_piece, QuantizedTensor)
                else rope_piece.to(device=device, dtype=dtype)
            )
            yield latent, rope
        if self.latent is not None and self.rope_key is not None and self.latent.shape[1]:
            yield (
                self.latent.to(device=device, dtype=dtype),
                self.rope_key.to(device=device, dtype=dtype),
            )

    def _encode(self, x: torch.Tensor, *, rope: bool = False) -> CachePiece:
        scheme = self.cache_dtype
        if scheme == "bfloat16" or (rope and not self.quantize_rope):
            return x.to(torch.bfloat16).contiguous()
        return quantize_tensor(x, scheme, group_size=self.group_size)

    def _flush_cold_chunks(self) -> None:
        if self.cache_dtype == "bfloat16" or self.latent is None or self.rope_key is None:
            return
        threshold = self.recent_tokens + self.chunk_tokens
        while self.latent.shape[1] > threshold:
            latent_chunk = self.latent[:, : self.chunk_tokens].contiguous()
            rope_chunk = self.rope_key[:, : self.chunk_tokens].contiguous()
            self.cold_latent.append(self._encode(latent_chunk))
            self.cold_rope.append(self._encode(rope_chunk, rope=True))
            self.latent = self.latent[:, self.chunk_tokens :].contiguous()
            self.rope_key = self.rope_key[:, self.chunk_tokens :].contiguous()

    def _replace_from_materialized(self, latent: torch.Tensor, rope_key: torch.Tensor) -> None:
        self.cold_latent.clear()
        self.cold_rope.clear()
        self.latent = latent.contiguous()
        self.rope_key = rope_key.contiguous()
        self._flush_cold_chunks()

    def append(
        self,
        latent: torch.Tensor,
        rope_key: torch.Tensor,
        window_size: int | None,
        sink_tokens: int,
    ) -> None:
        latent = latent.detach()
        rope_key = rope_key.detach()
        if self.latent is None:
            self.latent = latent
            self.rope_key = rope_key
        else:
            self.latent = torch.cat((self.latent, latent), dim=1)
            self.rope_key = torch.cat((self.rope_key, rope_key), dim=1)
        self._flush_cold_chunks()

        # Trim with chunk-sized hysteresis. Rebuilding a quantized 128K cache for
        # every generated token would destroy decode speed once the window is full.
        trim_slack = self.chunk_tokens if window_size is not None and window_size > 2 * self.chunk_tokens else 0
        if window_size is not None and self.length > window_size + trim_slack:
            all_latent = self.materialize_latent(dtype=latent.dtype, device=latent.device)
            all_rope = self.materialize_rope(dtype=rope_key.dtype, device=rope_key.device)
            assert all_latent is not None and all_rope is not None
            sink = min(sink_tokens, window_size)
            recent = window_size - sink
            if sink and recent:
                all_latent = torch.cat((all_latent[:, :sink], all_latent[:, -recent:]), dim=1)
                all_rope = torch.cat((all_rope[:, :sink], all_rope[:, -recent:]), dim=1)
            elif recent:
                all_latent = all_latent[:, -recent:]
                all_rope = all_rope[:, -recent:]
            else:
                all_latent = all_latent[:, :sink]
                all_rope = all_rope[:, :sink]
            self._replace_from_materialized(all_latent, all_rope)


@dataclass
class AsterCache:
    """Mixed cache: recurrent KDA state plus compressed latent-attention state."""

    seen_tokens: int = 0
    latent: dict[int, LatentLayerCache] = field(default_factory=dict)
    kda_states: dict[int, torch.Tensor] = field(default_factory=dict)
    fla_cache: Any | None = None
    cache_dtype: str = "bfloat16"
    cache_group_size: int = 64
    cache_recent_tokens: int = 512
    cache_chunk_tokens: int = 1024
    cache_quantize_rope: bool = False

    @classmethod
    def create(cls, use_fla: bool = False, config: AsterConfig | None = None) -> "AsterCache":
        kwargs: dict[str, Any] = {}
        if config is not None:
            kwargs = {
                "cache_dtype": config.cache_dtype,
                "cache_group_size": config.cache_group_size,
                "cache_recent_tokens": config.cache_recent_tokens,
                "cache_chunk_tokens": config.cache_chunk_tokens,
                "cache_quantize_rope": config.cache_quantize_rope,
            }
        cache = cls(**kwargs)
        if use_fla:
            try:
                from fla.models.utils import LegacyFLACache

                cache.fla_cache = LegacyFLACache()
            except Exception as exc:  # pragma: no cover - depends on optional CUDA package
                raise RuntimeError("FLA was requested but its cache could not be constructed") from exc
        return cache

    def latent_layer(self, layer_idx: int) -> LatentLayerCache:
        if layer_idx not in self.latent:
            self.latent[layer_idx] = LatentLayerCache(
                cache_dtype=self.cache_dtype,
                group_size=self.cache_group_size,
                recent_tokens=self.cache_recent_tokens,
                chunk_tokens=self.cache_chunk_tokens,
                quantize_rope=self.cache_quantize_rope,
            )
        return self.latent[layer_idx]

    @property
    def num_bytes(self) -> int:
        total = sum(layer.num_bytes for layer in self.latent.values())
        total += sum(state.numel() * state.element_size() for state in self.kda_states.values())
        total += _nested_tensor_bytes(self.fla_cache)
        return int(total)

    def reset(self) -> None:
        self.seen_tokens = 0
        self.latent.clear()
        self.kda_states.clear()
        if self.fla_cache is not None:
            self.fla_cache.reset()
