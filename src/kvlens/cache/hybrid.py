"""Hybrid KV cache with sliding and global layers."""

from __future__ import annotations

import torch

from kvlens.config import Gemma4Config


class HybridCache:
    """Per-layer cache that evicts only for sliding layers."""

    def __init__(self, config: Gemma4Config) -> None:
        self.config = config
        self.storage: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.last_evictions: dict[int, int] = {}
        self._cumulative_evictions: dict[int, int] = {}

    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int,
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del positions  # HybridCache uses FIFO; positions not needed
        cached = self.storage.get(layer_idx)
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

        self.storage[layer_idx] = (combined_key, combined_value)
        self.last_evictions[layer_idx] = evicted
        self._cumulative_evictions[layer_idx] = (
            self._cumulative_evictions.get(layer_idx, 0) + evicted
        )
        return combined_key, combined_value

    def get(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        return self.storage.get(layer_idx)

    def reset(self) -> None:
        self.storage.clear()
        self.last_evictions.clear()
        self._cumulative_evictions.clear()

    def key_offset(self, layer_idx: int) -> int:
        """Absolute position of the first cached key (accounts for evictions)."""
        return self._cumulative_evictions.get(layer_idx, 0)

    def seq_length(self, layer_idx: int) -> int:
        cached = self.storage.get(layer_idx)
        if cached is None:
            return 0
        return cached[0].shape[-2]

    def memory_bytes(self, layer_idx: int) -> int:
        cached = self.storage.get(layer_idx)
        if cached is None:
            return 0
        key, value = cached
        return (key.numel() * key.element_size()) + (value.numel() * value.element_size())
