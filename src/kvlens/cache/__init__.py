"""KV cache strategies for KVLens.

Four families of strategies are exposed:

- **Hybrid-adapted** (default): ``h2o``, ``snapkv``, ``streaming``, ``pyramidkv``.
  These check ``config.layer_params(layer_idx).window_size`` and skip eviction
  on global layers (where ``window_size`` is ``None``), keeping full context on
  the long-range layers. They are *not* faithful ports of the canonical papers
  — they are practical adaptations for hybrid sliding/global attention models.

- **Canonical baselines**: ``all_layers_h2o``, ``all_layers_snapkv``,
  ``all_layers_streaming``, ``all_layers_pyramidkv``. These force eviction on
  every layer (uniform per-layer budget). They reproduce the original paper
  formulations and serve as the reference that demonstrates why naive ports
  break on hybrid attention models.

- **Proportional baselines**: ``proportional_h2o``, ``proportional_snapkv``,
  ``proportional_streaming``, ``proportional_pyramidkv``. Iso-token across
  every layer — each layer keeps the same number of cached tokens, so byte
  share scales with the layer's per-token cost. On uniform-head_dim models
  this is identical to ``all_layers_*``; on heterogeneous models (Gemma 4)
  larger-head_dim layers receive proportionally more bytes. Defined only with
  ``--memory-budget`` (window_budget mode is undefined for these strategies).

- **Hybrid family**: ``hybrid_h2o``, ``hybrid_snapkv``, ``hybrid_streaming``.
  Sliding layers use ``HybridCache`` (FIFO down to the architectural window);
  global layers use the named sparse strategy with the entire memory budget
  allocated to them. Designed for narrow-window sliding/global models (Gemma 4)
  where the trained sliding pattern requires recency-preserving selection while
  global layers can absorb sparse-selection benefits.

Use ``standard`` for the full-cache reference. ``naive`` is full recomputation.
``quantized`` is orthogonal (compresses tokens, doesn't select them).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol, runtime_checkable

import torch

from kvlens.cache.allocation import allocate_memory_budget
from kvlens.cache.h2o import H2OCache
from kvlens.cache.hybrid import HybridCache
from kvlens.cache.layertype import LayerTypeAwareCache
from kvlens.cache.naive import NaiveCache
from kvlens.cache.pyramidkv import PyramidKVCache
from kvlens.cache.quantized import QuantizedCache
from kvlens.cache.snapkv import SnapKVCache
from kvlens.cache.streaming import StreamingLLMCache
from kvlens.config import Gemma4Config

_ALL_LAYERS_BASELINES: dict[str, tuple[type, dict[str, object]]] = {
    "all_layers_h2o": (H2OCache, {}),
    "all_layers_snapkv": (SnapKVCache, {}),
    "all_layers_streaming": (StreamingLLMCache, {}),
    "all_layers_pyramidkv": (PyramidKVCache, {"apply_to_all_layers": True}),
}

_PROPORTIONAL_BASELINES: dict[str, tuple[type, dict[str, object]]] = {
    "proportional_h2o": (H2OCache, {}),
    "proportional_snapkv": (SnapKVCache, {}),
    "proportional_streaming": (StreamingLLMCache, {}),
    "proportional_pyramidkv": (PyramidKVCache, {"apply_to_all_layers": True}),
}

_HYBRID_FAMILY: dict[str, str] = {
    "hybrid_h2o": "h2o",
    "hybrid_snapkv": "snapkv",
    "hybrid_streaming": "streaming",
}

_PYRAMIDKV_STRATEGIES = {"pyramidkv", "all_layers_pyramidkv", "proportional_pyramidkv"}


@runtime_checkable
class CacheProtocol(Protocol):
    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int,
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def get(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor] | None: ...

    def reset(self) -> None: ...

    def seq_length(self, layer_idx: int) -> int: ...

    def key_offset(self, layer_idx: int) -> int: ...

    def memory_bytes(self, layer_idx: int) -> int: ...


def create_cache(
    strategy: str,
    config: Gemma4Config,
    window_budget: int | None = None,
    memory_budget_bytes: int | None = None,
    dtype_bytes: int = 2,
) -> CacheProtocol:
    if window_budget is not None and memory_budget_bytes is not None:
        raise ValueError("Cannot pass both window_budget and memory_budget_bytes")

    precomputed: dict[int, int | None] | None = None
    if memory_budget_bytes is not None:
        precomputed = allocate_memory_budget(strategy, config, memory_budget_bytes, dtype_bytes)
        if precomputed:
            config = config.with_per_layer_budgets(precomputed)

    if strategy in _ALL_LAYERS_BASELINES:
        if window_budget is not None:
            sliding_params = replace(config.sliding, window_size=window_budget)
            global_params = replace(config.global_, window_size=window_budget)
            config = replace(config, sliding=sliding_params, global_=global_params)
        cls, kwargs = _ALL_LAYERS_BASELINES[strategy]
        if strategy in _PYRAMIDKV_STRATEGIES and precomputed is not None:
            return cls(config, precomputed_budgets=precomputed, **kwargs)
        return cls(config, **kwargs)

    if strategy in _PROPORTIONAL_BASELINES:
        if memory_budget_bytes is None:
            raise ValueError(
                f"strategy '{strategy}' requires --memory-budget (iso-token "
                "allocation is only defined under a total-memory budget)"
            )
        cls, kwargs = _PROPORTIONAL_BASELINES[strategy]
        if strategy in _PYRAMIDKV_STRATEGIES and precomputed is not None:
            return cls(config, precomputed_budgets=precomputed, **kwargs)
        return cls(config, **kwargs)

    if strategy in _HYBRID_FAMILY:
        # Globals get window_size from the per-layer budget applied above;
        # slidings keep their architectural window (allocator emits no override).
        if memory_budget_bytes is None:
            raise ValueError(
                f"strategy '{strategy}' requires --memory-budget (hybrid family is "
                "only defined under a total-memory budget)"
            )
        return LayerTypeAwareCache(
            config,
            sliding_strategy="standard",
            global_strategy=_HYBRID_FAMILY[strategy],
        )

    if window_budget is not None:
        sliding_params = replace(config.sliding, window_size=window_budget)
        config = replace(config, sliding=sliding_params)

    if strategy == "naive":
        return NaiveCache()
    if strategy == "standard":
        return HybridCache(config)
    if strategy == "quantized":
        return QuantizedCache(config)
    if strategy == "streaming":
        return StreamingLLMCache(config)
    if strategy == "h2o":
        return H2OCache(config)
    if strategy == "snapkv":
        return SnapKVCache(config)
    if strategy == "pyramidkv":
        if precomputed is not None:
            return PyramidKVCache(config, precomputed_budgets=precomputed)
        return PyramidKVCache(config)
    raise ValueError(f"unsupported cache strategy '{strategy}'")


__all__ = [
    "CacheProtocol",
    "H2OCache",
    "HybridCache",
    "LayerTypeAwareCache",
    "NaiveCache",
    "PyramidKVCache",
    "QuantizedCache",
    "SnapKVCache",
    "StreamingLLMCache",
    "create_cache",
]
