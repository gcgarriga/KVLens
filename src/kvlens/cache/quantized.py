"""Quantized KV cache with int8 storage."""

from __future__ import annotations

import torch

from kvlens.cache.hybrid import HybridCache
from kvlens.config import Gemma4Config


class QuantizedCache(HybridCache):
    """Hybrid cache that stores K/V tensors in int8 form internally."""

    def __init__(self, config: Gemma4Config) -> None:
        super().__init__(config)
        self.storage: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}  # type: ignore[assignment]  # stores (k_q, k_s, v_q, v_s) vs parent's (k, v)
        self.last_errors: dict[int, float] = {}
        self._cache_dtype: dict[int, torch.dtype] = {}

    def _quantize(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        max_abs = tensor.abs().max()
        scale = torch.where(max_abs == 0, torch.ones_like(max_abs), max_abs / 127.0)
        quantized = torch.clamp((tensor / scale).round(), -127, 127).to(torch.int8)
        return quantized, scale.to(dtype=torch.float32)

    def _dequantize(
        self,
        quantized: torch.Tensor,
        scale: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return quantized.to(torch.float32).mul(scale).to(dtype=dtype)

    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int,
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del positions  # QuantizedCache uses FIFO
        cached = self.get(layer_idx)
        if cached is None:
            combined_key = key
            combined_value = value
        else:
            combined_key = torch.cat((cached[0], key), dim=-2)
            combined_value = torch.cat((cached[1], value), dim=-2)

        window_size = self.config.layer_params(layer_idx).window_size
        evicted = 0
        if window_size is not None and combined_key.shape[-2] > window_size:
            evicted = combined_key.shape[-2] - window_size
            combined_key = combined_key[:, :, -window_size:, :]
            combined_value = combined_value[:, :, -window_size:, :]

        quantized_key, key_scale = self._quantize(combined_key)
        quantized_value, value_scale = self._quantize(combined_value)
        dequantized_key = self._dequantize(quantized_key, key_scale, combined_key.dtype)
        dequantized_value = self._dequantize(quantized_value, value_scale, combined_value.dtype)

        self.storage[layer_idx] = (quantized_key, key_scale, quantized_value, value_scale)
        self._cache_dtype[layer_idx] = combined_key.dtype
        self.last_evictions[layer_idx] = evicted
        self._cumulative_evictions[layer_idx] = (
            self._cumulative_evictions.get(layer_idx, 0) + evicted
        )
        self.last_errors[layer_idx] = float(
            torch.linalg.vector_norm(combined_key - dequantized_key).item()
            + torch.linalg.vector_norm(combined_value - dequantized_value).item()
        )
        return dequantized_key, dequantized_value

    def get(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        cached = self.storage.get(layer_idx)
        if cached is None:
            return None
        quantized_key, key_scale, quantized_value, value_scale = cached
        dtype = self._cache_dtype.get(layer_idx, torch.float32)
        return (
            self._dequantize(quantized_key, key_scale, dtype),
            self._dequantize(quantized_value, value_scale, dtype),
        )

    def seq_length(self, layer_idx: int) -> int:
        cached = self.storage.get(layer_idx)
        if cached is None:
            return 0
        return cached[0].shape[-2]

    def reset(self) -> None:
        super().reset()
        self.last_errors.clear()
        self._cache_dtype.clear()

    def memory_bytes(self, layer_idx: int) -> int:
        cached = self.storage.get(layer_idx)
        if cached is None:
            return 0
        quantized_key, key_scale, quantized_value, value_scale = cached
        return (
            (quantized_key.numel() * quantized_key.element_size())
            + (quantized_value.numel() * quantized_value.element_size())
            + (key_scale.numel() * key_scale.element_size())
            + (value_scale.numel() * value_scale.element_size())
        )
