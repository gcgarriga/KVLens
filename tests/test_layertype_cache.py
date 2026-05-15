"""Tests for LayerTypeAwareCache."""

from __future__ import annotations

import pytest
import torch

from kvlens.cache import create_cache
from kvlens.cache.hybrid import HybridCache
from kvlens.cache.layertype import LayerTypeAwareCache
from kvlens.cache.streaming import StreamingLLMCache
from kvlens.config import Gemma4Config


@pytest.fixture
def tiny_config() -> Gemma4Config:
    return Gemma4Config.tiny()


def test_sliding_layers_routed_to_sliding_cache(tiny_config) -> None:
    cache = LayerTypeAwareCache(
        tiny_config, sliding_strategy="streaming", global_strategy="standard"
    )
    sliding_idx = 0
    assert tiny_config.layer_types[sliding_idx] == "sliding"

    k = torch.randn(1, tiny_config.num_kv_heads, 10, 32)
    v = torch.randn(1, tiny_config.num_kv_heads, 10, 32)
    cache.update(k, v, sliding_idx, torch.arange(10))

    assert isinstance(cache._sliding_cache, StreamingLLMCache)
    assert cache._sliding_cache.seq_length(sliding_idx) == 10


def test_global_layers_routed_to_global_cache(tiny_config) -> None:
    cache = LayerTypeAwareCache(tiny_config, sliding_strategy="h2o", global_strategy="standard")
    global_idx = 3
    assert tiny_config.layer_types[global_idx] == "global"

    k = torch.randn(1, tiny_config.num_kv_heads, 50, 64)
    v = torch.randn(1, tiny_config.num_kv_heads, 50, 64)
    cache.update(k, v, global_idx, torch.arange(50))

    assert isinstance(cache._global_cache, HybridCache)
    assert cache._global_cache.seq_length(global_idx) == 50


def test_reset_clears_both_sub_caches(tiny_config) -> None:
    cache = LayerTypeAwareCache(tiny_config, sliding_strategy="h2o", global_strategy="standard")
    k = torch.randn(1, tiny_config.num_kv_heads, 5, 32)
    v = torch.randn(1, tiny_config.num_kv_heads, 5, 32)
    cache.update(k, v, 0, torch.arange(5))

    cache.reset()
    assert cache.seq_length(0) == 0


def test_sliding_budget_overrides_window_size(tiny_config) -> None:
    cache = LayerTypeAwareCache(
        tiny_config, sliding_strategy="standard", global_strategy="standard", sliding_budget=16
    )
    sliding_idx = 0
    k = torch.randn(1, tiny_config.num_kv_heads, 20, 32)
    v = torch.randn(1, tiny_config.num_kv_heads, 20, 32)
    k_out, _ = cache.update(k, v, sliding_idx)
    assert k_out.shape[-2] == 16


def test_update_scores_forwarded_to_sliding_cache(tiny_config) -> None:
    cache = LayerTypeAwareCache(tiny_config, sliding_strategy="h2o", global_strategy="standard")
    sliding_idx = 0
    k = torch.randn(1, tiny_config.num_kv_heads, 4, 32)
    v = torch.randn(1, tiny_config.num_kv_heads, 4, 32)
    cache.update(k, v, sliding_idx)

    attn = torch.softmax(torch.randn(1, tiny_config.num_heads, 1, 4), dim=-1)
    cache.update_scores(sliding_idx, attn)

    assert sliding_idx in cache._sliding_cache.scores


def test_create_cache_window_budget_override(tiny_config) -> None:
    cache = create_cache("standard", tiny_config, window_budget=16)
    k = torch.randn(1, tiny_config.num_kv_heads, 20, 32)
    v = torch.randn(1, tiny_config.num_kv_heads, 20, 32)
    k_out, _ = cache.update(k, v, layer_idx=0)
    assert k_out.shape[-2] == 16


def test_cache_positions_forwarded_from_streaming_sliding_cache(tiny_config) -> None:
    """When sliding_strategy='streaming', cache_positions() must propagate up."""
    cache = LayerTypeAwareCache(
        tiny_config, sliding_strategy="streaming", global_strategy="standard"
    )
    sliding_idx = 0
    window = tiny_config.layer_params(sliding_idx).window_size  # 64

    k = torch.randn(1, tiny_config.num_kv_heads, window + 5, 32)
    v = torch.randn(1, tiny_config.num_kv_heads, window + 5, 32)
    cache.update(k, v, sliding_idx, torch.arange(window + 5))

    pos = cache.cache_positions(sliding_idx)
    assert pos is not None
    assert pos.shape[0] == window
    for i in range(4):
        assert pos[i].item() == i


def test_cache_positions_none_for_standard_sliding(tiny_config) -> None:
    cache = LayerTypeAwareCache(
        tiny_config, sliding_strategy="standard", global_strategy="standard"
    )
    k = torch.randn(1, tiny_config.num_kv_heads, 5, 32)
    v = torch.randn(1, tiny_config.num_kv_heads, 5, 32)
    cache.update(k, v, layer_idx=0)
    assert cache.cache_positions(0) is None


def test_memory_bytes_delegated_per_layer_type(tiny_config) -> None:
    cache = LayerTypeAwareCache(tiny_config, sliding_strategy="h2o", global_strategy="standard")
    sliding_idx = 0
    global_idx = 3

    k_s = torch.randn(1, tiny_config.num_kv_heads, 10, 32)
    v_s = torch.randn(1, tiny_config.num_kv_heads, 10, 32)
    cache.update(k_s, v_s, sliding_idx)

    k_g = torch.randn(1, tiny_config.num_kv_heads, 20, 64)
    v_g = torch.randn(1, tiny_config.num_kv_heads, 20, 64)
    cache.update(k_g, v_g, global_idx)

    sliding_bytes = cache.memory_bytes(sliding_idx)
    global_bytes = cache.memory_bytes(global_idx)
    assert sliding_bytes > 0
    assert global_bytes > 0
    assert global_bytes > sliding_bytes


def test_last_evictions_merged_across_sub_caches(tiny_config) -> None:
    cache = LayerTypeAwareCache(
        tiny_config, sliding_strategy="standard", global_strategy="standard"
    )
    sliding_idx = 0
    global_idx = 3

    k_s = torch.randn(1, tiny_config.num_kv_heads, 70, 32)
    v_s = torch.randn(1, tiny_config.num_kv_heads, 70, 32)
    cache.update(k_s, v_s, sliding_idx)

    k_g = torch.randn(1, tiny_config.num_kv_heads, 20, 64)
    v_g = torch.randn(1, tiny_config.num_kv_heads, 20, 64)
    cache.update(k_g, v_g, global_idx)

    assert cache.last_evictions[sliding_idx] == 6
    assert cache.last_evictions[global_idx] == 0


def test_update_scores_on_global_layer_is_no_op(tiny_config) -> None:
    """update_scores on a global layer with standard sub-cache must not crash."""
    cache = LayerTypeAwareCache(tiny_config, sliding_strategy="h2o", global_strategy="standard")
    global_idx = 3
    k = torch.randn(1, tiny_config.num_kv_heads, 10, 64)
    v = torch.randn(1, tiny_config.num_kv_heads, 10, 64)
    cache.update(k, v, global_idx)

    attn = torch.softmax(torch.randn(1, tiny_config.num_heads, 1, 10), dim=-1)
    cache.update_scores(global_idx, attn)  # must not raise


def test_layertype_h2o_keeps_global_context_unlike_all_layers_h2o(tiny_config) -> None:
    global_idx = 3
    key = torch.randn(1, tiny_config.num_kv_heads, 20, 64)
    value = torch.randn(1, tiny_config.num_kv_heads, 20, 64)

    layertype = LayerTypeAwareCache(
        tiny_config, sliding_strategy="h2o", global_strategy="standard", sliding_budget=8
    )
    all_layers = create_cache("all_layers_h2o", tiny_config, window_budget=8)

    layertype_key, _ = layertype.update(key, value, global_idx, positions=torch.arange(20))
    all_layers_key, _ = all_layers.update(key, value, global_idx, positions=torch.arange(20))

    assert layertype_key.shape[-2] == 20
    assert layertype.key_offset(global_idx) == 0
    assert all_layers_key.shape[-2] == 8
    assert all_layers.key_offset(global_idx) == 12


def test_create_cache_hybrid_h2o_routes_sliding_fifo_global_h2o(tiny_config) -> None:
    from kvlens.cache.h2o import H2OCache

    cache = create_cache("hybrid_h2o", tiny_config, memory_budget_bytes=4096)
    assert isinstance(cache, LayerTypeAwareCache)
    assert isinstance(cache._sliding_cache, HybridCache)
    assert not isinstance(cache._sliding_cache, H2OCache)  # FIFO, not score-based
    assert isinstance(cache._global_cache, H2OCache)


def test_hybrid_h2o_global_layer_evicts_when_budget_binding(tiny_config) -> None:
    cache = create_cache("hybrid_h2o", tiny_config, memory_budget_bytes=512)
    global_idx = next(
        i for i in range(tiny_config.num_layers) if tiny_config.layer_types[i] == "global"
    )
    head_dim = tiny_config.layer_params(global_idx).head_dim
    k = torch.randn(1, tiny_config.num_kv_heads, 80, head_dim)
    v = torch.randn(1, tiny_config.num_kv_heads, 80, head_dim)
    k_out, _ = cache.update(k, v, global_idx, torch.arange(80))
    assert k_out.shape[-2] < 80, "global layer must evict under tight budget"
    assert cache.key_offset(global_idx) > 0


def test_hybrid_h2o_sliding_layer_uses_architectural_window(tiny_config) -> None:
    arch_window = tiny_config.layer_params(0).window_size  # tiny config sliding window = 64
    assert tiny_config.layer_types[0] == "sliding"
    # 4 MB budget — non-binding; sliding layer should keep its architectural window
    cache = create_cache("hybrid_h2o", tiny_config, memory_budget_bytes=4 << 20)
    head_dim = tiny_config.layer_params(0).head_dim
    k = torch.randn(1, tiny_config.num_kv_heads, arch_window + 20, head_dim)
    v = torch.randn(1, tiny_config.num_kv_heads, arch_window + 20, head_dim)
    k_out, _ = cache.update(k, v, 0, torch.arange(arch_window + 20))
    assert k_out.shape[-2] == arch_window, "sliding layer must FIFO down to architectural window"


def test_create_cache_hybrid_snapkv_routes_sliding_fifo_global_snapkv(tiny_config) -> None:
    from kvlens.cache.snapkv import SnapKVCache

    cache = create_cache("hybrid_snapkv", tiny_config, memory_budget_bytes=4096)
    assert isinstance(cache, LayerTypeAwareCache)
    assert isinstance(cache._sliding_cache, HybridCache)
    assert isinstance(cache._global_cache, SnapKVCache)


def test_hybrid_snapkv_accumulate_prefill_scores_routes_to_global(tiny_config) -> None:
    cache = create_cache("hybrid_snapkv", tiny_config, memory_budget_bytes=4096)
    global_idx = 3
    head_dim = tiny_config.layer_params(global_idx).head_dim
    k = torch.randn(1, tiny_config.num_kv_heads, 50, head_dim)
    v = torch.randn(1, tiny_config.num_kv_heads, 50, head_dim)
    cache.update(k, v, global_idx, torch.arange(50))
    attn = torch.softmax(torch.randn(1, tiny_config.num_heads, 50, 50), dim=-1)
    cache.accumulate_prefill_scores(global_idx, attn)
    assert global_idx in cache._global_cache._prefill_scores


def test_create_cache_hybrid_streaming_routes_sliding_fifo_global_streaming(tiny_config) -> None:
    from kvlens.cache.streaming import StreamingLLMCache

    cache = create_cache("hybrid_streaming", tiny_config, memory_budget_bytes=4096)
    assert isinstance(cache, LayerTypeAwareCache)
    assert isinstance(cache._sliding_cache, HybridCache)
    assert isinstance(cache._global_cache, StreamingLLMCache)


def test_hybrid_streaming_global_layer_keeps_sinks_on_eviction(tiny_config) -> None:
    # 4096 bytes / (1 global layer × 512 bytes-per-token) = 8-token budget,
    # leaving room for 4 sinks + 4 recent under StreamingLLMCache's policy.
    cache = create_cache("hybrid_streaming", tiny_config, memory_budget_bytes=4096)
    global_idx = 3
    head_dim = tiny_config.layer_params(global_idx).head_dim
    k = torch.randn(1, tiny_config.num_kv_heads, 80, head_dim)
    v = torch.randn(1, tiny_config.num_kv_heads, 80, head_dim)
    cache.update(k, v, global_idx, torch.arange(80))
    pos = cache.cache_positions(global_idx)
    assert pos is not None
    # First 4 sink positions must remain in the cached set
    for sink_pos in range(4):
        assert sink_pos in pos.tolist()
