"""Tests for PyramidKVCache."""

from __future__ import annotations

import pytest
import torch

from kvlens.cache.pyramidkv import PyramidKVCache
from kvlens.config import Gemma4Config


@pytest.fixture
def tiny_config() -> Gemma4Config:
    return Gemma4Config.tiny()


class TestPyramidKVCache:
    def test_shallower_sliding_layers_get_larger_budget(self, tiny_config) -> None:
        """Sliding layer budgets must decrease monotonically with depth."""
        cache = PyramidKVCache(tiny_config, min_budget=8)
        sliding_indices = [i for i, t in enumerate(tiny_config.layer_types) if t == "sliding"]
        assert len(sliding_indices) >= 2, "tiny_config needs at least 2 sliding layers"
        budgets = [cache._budgets[i] for i in sliding_indices]
        for earlier, later in zip(budgets, budgets[1:], strict=False):
            assert earlier >= later, f"budget should not increase with depth: {budgets}"

    def test_global_layers_have_no_budget_cap(self, tiny_config) -> None:
        cache = PyramidKVCache(tiny_config, min_budget=8)
        global_indices = [i for i, t in enumerate(tiny_config.layer_types) if t == "global"]
        for idx in global_indices:
            assert cache._budgets[idx] is None

    def test_global_layers_not_evicted(self, tiny_config) -> None:
        cache = PyramidKVCache(tiny_config, min_budget=8)
        global_idx = next(i for i, t in enumerate(tiny_config.layer_types) if t == "global")
        k = torch.randn(1, tiny_config.num_kv_heads, 200, 64)
        v = torch.randn(1, tiny_config.num_kv_heads, 200, 64)
        k_out, _ = cache.update(k, v, global_idx)
        assert k_out.shape[-2] == 200

    def test_deepest_sliding_layer_evicts_to_min_budget(self, tiny_config) -> None:
        cache = PyramidKVCache(tiny_config, min_budget=8)
        sliding_indices = [i for i, t in enumerate(tiny_config.layer_types) if t == "sliding"]
        deepest = sliding_indices[-1]
        budget = cache._budgets[deepest]
        assert budget == 8

        k = torch.randn(1, tiny_config.num_kv_heads, budget + 10, 32)
        v = torch.randn(1, tiny_config.num_kv_heads, budget + 10, 32)
        k_out, _ = cache.update(k, v, deepest)
        assert k_out.shape[-2] == budget

    def test_shallowest_sliding_layer_keeps_full_window(self, tiny_config) -> None:
        cache = PyramidKVCache(tiny_config, min_budget=8)
        sliding_indices = [i for i, t in enumerate(tiny_config.layer_types) if t == "sliding"]
        shallowest = sliding_indices[0]
        expected_budget = tiny_config.layer_params(shallowest).window_size
        assert cache._budgets[shallowest] == expected_budget

    def test_reset_clears_storage(self, tiny_config) -> None:
        cache = PyramidKVCache(tiny_config, min_budget=8)
        sliding_idx = next(i for i, t in enumerate(tiny_config.layer_types) if t == "sliding")
        k = torch.randn(1, tiny_config.num_kv_heads, 5, 32)
        v = torch.randn(1, tiny_config.num_kv_heads, 5, 32)
        cache.update(k, v, sliding_idx)
        cache.reset()
        assert cache.seq_length(sliding_idx) == 0

    def test_memory_bytes_nonzero_after_update(self, tiny_config) -> None:
        cache = PyramidKVCache(tiny_config, min_budget=8)
        sliding_idx = next(i for i, t in enumerate(tiny_config.layer_types) if t == "sliding")
        k = torch.randn(1, tiny_config.num_kv_heads, 5, 32)
        v = torch.randn(1, tiny_config.num_kv_heads, 5, 32)
        cache.update(k, v, sliding_idx)
        assert cache.memory_bytes(sliding_idx) > 0

    def test_default_min_budget_scales_with_low_window_budget(self, tiny_config) -> None:
        from kvlens.cache import create_cache

        cache = create_cache("pyramidkv", tiny_config, window_budget=8)
        sliding_indices = [i for i, t in enumerate(tiny_config.layer_types) if t == "sliding"]
        budgets = [cache._budgets[i] for i in sliding_indices]

        assert budgets[0] == 8
        assert budgets[-1] == 2
        assert all(b is not None and b <= 8 for b in budgets)

    def test_explicit_min_budget_is_capped_to_window(self, tiny_config) -> None:
        from dataclasses import replace

        sliding = replace(tiny_config.sliding, window_size=8)
        config = replace(tiny_config, sliding=sliding)
        cache = PyramidKVCache(config, min_budget=64)
        sliding_indices = [i for i, t in enumerate(config.layer_types) if t == "sliding"]

        assert all(cache._budgets[i] == 8 for i in sliding_indices)
