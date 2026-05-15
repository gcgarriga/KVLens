"""LayerTypeAwareCache: different eviction policies for sliding vs global layers."""

from __future__ import annotations

import torch

from kvlens.config import Gemma4Config


class LayerTypeAwareCache:
    """Routes each layer to a sliding-layer cache or global-layer cache.

    At the same total memory budget, applies a smarter eviction policy on
    sliding layers while keeping global layers at full context, instead of
    applying one policy uniformly.
    """

    def __init__(
        self,
        config: Gemma4Config,
        sliding_strategy: str,
        global_strategy: str,
        sliding_budget: int | None = None,
    ) -> None:
        from kvlens.cache import create_cache

        # Both sub-caches receive `config` unchanged. Callers that pre-apply
        # `with_per_layer_budgets` (e.g. the hybrid_* factory branch) rely on
        # those overrides reaching the sub-caches via this same reference.
        self.config = config
        self._sliding_cache = create_cache(sliding_strategy, config, window_budget=sliding_budget)
        self._global_cache = create_cache(global_strategy, config, window_budget=None)

    def _cache_for(self, layer_idx: int):
        if self.config.layer_types[layer_idx] == "global":
            return self._global_cache
        return self._sliding_cache

    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int,
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._cache_for(layer_idx).update(key, value, layer_idx, positions)

    def get(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        return self._cache_for(layer_idx).get(layer_idx)

    def reset(self) -> None:
        self._sliding_cache.reset()
        self._global_cache.reset()

    def seq_length(self, layer_idx: int) -> int:
        return self._cache_for(layer_idx).seq_length(layer_idx)

    def key_offset(self, layer_idx: int) -> int:
        return self._cache_for(layer_idx).key_offset(layer_idx)

    def memory_bytes(self, layer_idx: int) -> int:
        return self._cache_for(layer_idx).memory_bytes(layer_idx)

    @property
    def last_evictions(self) -> dict[int, int]:
        evictions: dict[int, int] = {}
        if hasattr(self._sliding_cache, "last_evictions"):
            evictions.update(self._sliding_cache.last_evictions)
        if hasattr(self._global_cache, "last_evictions"):
            evictions.update(self._global_cache.last_evictions)
        return evictions

    def update_scores(self, layer_idx: int, attention_weights: torch.Tensor) -> None:
        cache = self._cache_for(layer_idx)
        if hasattr(cache, "update_scores"):
            cache.update_scores(layer_idx, attention_weights)

    def accumulate_prefill_scores(self, layer_idx: int, attention_weights: torch.Tensor) -> None:
        cache = self._cache_for(layer_idx)
        if hasattr(cache, "accumulate_prefill_scores"):
            cache.accumulate_prefill_scores(layer_idx, attention_weights)

    def cache_positions(self, layer_idx: int) -> torch.Tensor | None:
        cache = self._cache_for(layer_idx)
        if hasattr(cache, "cache_positions"):
            return cache.cache_positions(layer_idx)
        return None
