"""Tests for cache strategies that retain non-contiguous token positions."""

from __future__ import annotations

import torch

from kvlens.cache.h2o import H2OCache
from kvlens.cache.snapkv import SnapKVCache
from kvlens.config import Gemma4Config


def test_h2o_exposes_positions_after_noncontiguous_eviction(
    tiny_config: Gemma4Config,
) -> None:
    cache = H2OCache(tiny_config)
    layer_idx = 0
    window = tiny_config.layer_params(layer_idx).window_size
    assert window is not None
    key = torch.arange(window + 2, dtype=torch.float32).view(1, 1, window + 2, 1)
    value = key.clone()
    positions = torch.arange(window + 2)
    scores = torch.ones(window + 2, dtype=torch.float32)
    scores[2:4] = 0.0
    cache.scores[layer_idx] = scores

    cache.update(key, value, layer_idx, positions=positions)

    cached_positions = cache.cache_positions(layer_idx)
    assert cached_positions is not None
    assert cached_positions.tolist() == [0, 1, *range(4, window + 2)]


def test_snapkv_exposes_positions_after_noncontiguous_snap(
    tiny_config: Gemma4Config,
) -> None:
    cache = SnapKVCache(tiny_config)
    layer_idx = 0
    window = tiny_config.layer_params(layer_idx).window_size
    assert window is not None
    key = torch.arange(window + 2, dtype=torch.float32).view(1, 1, window + 2, 1)
    value = key.clone()
    positions = torch.arange(window + 2)
    scores = torch.ones(window + 2, dtype=torch.float32)
    scores[2:4] = 0.0

    cache.update(key, value, layer_idx, positions=positions)
    cache._prefill_scores[layer_idx] = scores
    cache._snap(layer_idx)

    cached_positions = cache.cache_positions(layer_idx)
    assert cached_positions is not None
    assert cached_positions.tolist() == [0, 1, *range(4, window + 2)]
