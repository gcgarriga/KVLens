"""Tests for cache strategies."""

from __future__ import annotations

import pytest
import torch

from kvlens.cache import CacheProtocol, create_cache
from kvlens.cache.h2o import H2OCache
from kvlens.cache.hybrid import HybridCache
from kvlens.cache.naive import NaiveCache
from kvlens.cache.quantized import QuantizedCache
from kvlens.cache.streaming import StreamingLLMCache
from kvlens.config import Gemma4Config
from kvlens.generation import teacher_forced_logits_with_cache
from kvlens.model import GemmaModel


class TestNaiveCache:
    def test_returns_inputs_and_stores_nothing(self) -> None:
        cache = NaiveCache()
        key = torch.randn(1, 2, 3, 4)
        value = torch.randn(1, 2, 3, 4)

        cached_key, cached_value = cache.update(key, value, layer_idx=0)

        torch.testing.assert_close(cached_key, key)
        torch.testing.assert_close(cached_value, value)
        assert cache.get(0) is None
        assert cache.seq_length(0) == 0


class TestHybridCache:
    def test_sliding_layers_evict_beyond_window(self, tiny_config) -> None:
        cache = HybridCache(tiny_config)
        key = torch.randn(1, tiny_config.num_kv_heads, 70, 32)
        value = torch.randn(1, tiny_config.num_kv_heads, 70, 32)

        cached_key, cached_value = cache.update(key, value, layer_idx=0)

        assert cached_key.shape[-2] == 64
        assert cached_value.shape[-2] == 64
        assert cache.last_evictions[0] == 6

    def test_global_layers_keep_full_history(self, tiny_config) -> None:
        cache = HybridCache(tiny_config)
        key = torch.randn(1, tiny_config.num_kv_heads, 70, 64)
        value = torch.randn(1, tiny_config.num_kv_heads, 70, 64)

        cached_key, cached_value = cache.update(key, value, layer_idx=3)

        assert cached_key.shape[-2] == 70
        assert cached_value.shape[-2] == 70
        assert cache.last_evictions[3] == 0

    def test_hybrid_cache_matches_full_recompute_logits(self, tiny_config) -> None:
        model = GemmaModel(tiny_config)
        input_ids = torch.tensor([[1, 2, 3, 4]])

        full_recompute = model(input_ids[:, :-1])
        cached = teacher_forced_logits_with_cache(model, input_ids, HybridCache(tiny_config))

        torch.testing.assert_close(cached, full_recompute, atol=1e-5, rtol=1e-5)


class TestQuantizedCache:
    def test_quantize_dequantize_error_stays_small(self, tiny_config) -> None:
        cache = QuantizedCache(tiny_config)
        key = torch.randn(1, tiny_config.num_kv_heads, 6, 32)
        value = torch.randn(1, tiny_config.num_kv_heads, 6, 32)

        dequantized_key, dequantized_value = cache.update(key, value, layer_idx=0)

        assert cache.last_errors[0] < 1.0
        assert dequantized_key.shape == key.shape
        assert dequantized_value.shape == value.shape

    def test_preserves_input_dtype(self, tiny_config) -> None:
        cache = QuantizedCache(tiny_config)
        key = torch.randn(1, tiny_config.num_kv_heads, 4, 32, dtype=torch.bfloat16)
        value = torch.randn(1, tiny_config.num_kv_heads, 4, 32, dtype=torch.bfloat16)

        k_out, v_out = cache.update(key, value, layer_idx=0)
        assert k_out.dtype == torch.bfloat16
        assert v_out.dtype == torch.bfloat16

        k_get, v_get = cache.get(0)
        assert k_get.dtype == torch.bfloat16
        assert v_get.dtype == torch.bfloat16

    def test_eviction_tracked_via_key_offset(self, tiny_config) -> None:
        """Regression: QuantizedCache must keep _cumulative_evictions in sync."""
        cache = QuantizedCache(tiny_config)
        # Find a sliding layer
        sliding_idx = next(i for i, t in enumerate(tiny_config.layer_types) if t == "sliding")
        window = tiny_config.layer_params(sliding_idx).window_size
        assert window is not None

        k1 = torch.randn(1, tiny_config.num_kv_heads, window, 32)
        v1 = torch.randn(1, tiny_config.num_kv_heads, window, 32)
        cache.update(k1, v1, sliding_idx)
        assert cache.key_offset(sliding_idx) == 0

        k2 = torch.randn(1, tiny_config.num_kv_heads, 5, 32)
        v2 = torch.randn(1, tiny_config.num_kv_heads, 5, 32)
        cache.update(k2, v2, sliding_idx)
        assert cache.key_offset(sliding_idx) == 5
        assert cache.seq_length(sliding_idx) == window

    def test_reset_clears_dtype_and_errors(self, tiny_config) -> None:
        cache = QuantizedCache(tiny_config)
        key = torch.randn(1, tiny_config.num_kv_heads, 4, 32, dtype=torch.bfloat16)
        value = torch.randn(1, tiny_config.num_kv_heads, 4, 32, dtype=torch.bfloat16)
        cache.update(key, value, layer_idx=0)

        cache.reset()
        assert cache.last_errors == {}
        assert cache._cache_dtype == {}
        assert cache.key_offset(0) == 0


class TestCacheProtocol:
    @pytest.mark.parametrize(
        "strategy",
        [
            "naive",
            "standard",
            "quantized",
            "streaming",
            "h2o",
            "snapkv",
            "pyramidkv",
            "all_layers_h2o",
            "all_layers_snapkv",
            "all_layers_streaming",
            "all_layers_pyramidkv",
        ],
    )
    def test_cache_implements_protocol(self, tiny_config: Gemma4Config, strategy: str) -> None:
        cache = create_cache(strategy, tiny_config)
        assert isinstance(cache, CacheProtocol)

    @pytest.mark.parametrize(
        "strategy",
        ["standard", "streaming", "h2o", "snapkv", "pyramidkv"],
    )
    def test_sliding_strategy_respects_low_window_budget(
        self, tiny_config: Gemma4Config, strategy: str
    ) -> None:
        cache = create_cache(strategy, tiny_config, window_budget=8)
        key = torch.randn(1, tiny_config.num_kv_heads, 20, 32)
        value = torch.randn(1, tiny_config.num_kv_heads, 20, 32)
        cached_key, _ = cache.update(key, value, layer_idx=0, positions=torch.arange(20))
        if strategy == "snapkv":
            # SnapKV keeps full prefill until the first decode token.
            decode_key = torch.randn(1, tiny_config.num_kv_heads, 1, 32)
            decode_value = torch.randn(1, tiny_config.num_kv_heads, 1, 32)
            cached_key, _ = cache.update(
                decode_key, decode_value, layer_idx=0, positions=torch.tensor([20])
            )
        assert cached_key.shape[-2] <= 8
        assert cache.seq_length(0) <= 8
        assert cache.memory_bytes(0) >= 0

    def test_all_layers_h2o_evicts_global_layers(self, tiny_config: Gemma4Config) -> None:
        cache = create_cache("all_layers_h2o", tiny_config, window_budget=8)
        global_idx = 3
        key = torch.randn(1, tiny_config.num_kv_heads, 20, 64)
        value = torch.randn(1, tiny_config.num_kv_heads, 20, 64)

        cached_key, _ = cache.update(key, value, global_idx, positions=torch.arange(20))

        assert cached_key.shape[-2] == 8
        assert cache.seq_length(global_idx) == 8
        assert cache.key_offset(global_idx) == 12
        assert cache.last_evictions[global_idx] == 12


class TestStreamingLLMCache:
    def test_preserves_sink_tokens_over_oldest(self, tiny_config) -> None:
        cache = StreamingLLMCache(tiny_config, sink_tokens=4)
        sliding_idx = 0  # sliding layer, window=64
        window = tiny_config.layer_params(sliding_idx).window_size  # 64

        # Create keys where first 4 are distinctively marked
        k = torch.zeros(1, tiny_config.num_kv_heads, window + 10, 32)
        v = torch.zeros(1, tiny_config.num_kv_heads, window + 10, 32)
        for i in range(4):
            k[0, :, i, 0] = float(i + 100)  # sinks: 100,101,102,103
        positions = torch.arange(window + 10)
        k_out, v_out = cache.update(k, v, sliding_idx, positions)

        assert k_out.shape[-2] == window
        # First 4 tokens in output should be the sinks
        for i in range(4):
            assert (k_out[0, 0, i, 0] == float(i + 100)).item()

    def test_recency_window_fills_remaining_slots(self, tiny_config) -> None:
        cache = StreamingLLMCache(tiny_config, sink_tokens=4)
        sliding_idx = 0
        window = tiny_config.layer_params(sliding_idx).window_size  # 64

        k = torch.zeros(1, tiny_config.num_kv_heads, window + 10, 32)
        v = torch.zeros(1, tiny_config.num_kv_heads, window + 10, 32)
        for i in range(window + 10):
            k[0, :, i, 0] = float(i)
        positions = torch.arange(window + 10)
        k_out, _ = cache.update(k, v, sliding_idx, positions)

        assert k_out.shape[-2] == window
        # Last (window - 4) tokens should be the most recent
        recent_size = window - 4
        for j in range(recent_size):
            expected = float((window + 10) - recent_size + j)
            assert (k_out[0, 0, 4 + j, 0] == expected).item()

    def test_global_layers_unaffected(self, tiny_config) -> None:
        cache = StreamingLLMCache(tiny_config, sink_tokens=4)
        global_idx = 3  # global layer in tiny config (window_size=None)

        k = torch.randn(1, tiny_config.num_kv_heads, 100, 64)
        v = torch.randn(1, tiny_config.num_kv_heads, 100, 64)
        positions = torch.arange(100)
        k_out, v_out = cache.update(k, v, global_idx, positions)

        assert k_out.shape[-2] == 100  # no eviction

    def test_cache_positions_exposed_for_mask(self, tiny_config) -> None:
        cache = StreamingLLMCache(tiny_config, sink_tokens=4)
        sliding_idx = 0
        window = tiny_config.layer_params(sliding_idx).window_size  # 64

        k = torch.zeros(1, tiny_config.num_kv_heads, window + 5, 32)
        v = torch.zeros(1, tiny_config.num_kv_heads, window + 5, 32)
        positions = torch.arange(window + 5)
        cache.update(k, v, sliding_idx, positions)

        cached_pos = cache.cache_positions(sliding_idx)
        assert cached_pos is not None
        assert cached_pos.shape[0] == window
        # Sinks at 0-3, then recent positions
        for i in range(4):
            assert cached_pos[i].item() == i
        assert cached_pos[-1].item() == window + 4  # last position

    def test_incremental_decode_updates_positions(self, tiny_config) -> None:
        cache = StreamingLLMCache(tiny_config, sink_tokens=4)
        sliding_idx = 0
        window = tiny_config.layer_params(sliding_idx).window_size  # 64

        # Prefill with window tokens
        k = torch.randn(1, tiny_config.num_kv_heads, window, 32)
        v = torch.randn(1, tiny_config.num_kv_heads, window, 32)
        cache.update(k, v, sliding_idx, torch.arange(window))

        # Decode one more token
        k2 = torch.randn(1, tiny_config.num_kv_heads, 1, 32)
        v2 = torch.randn(1, tiny_config.num_kv_heads, 1, 32)
        cache.update(k2, v2, sliding_idx, torch.tensor([window]))

        pos = cache.cache_positions(sliding_idx)
        assert pos is not None
        assert pos.shape[0] == window
        assert pos[0].item() == 0  # sink preserved
        assert pos[-1].item() == window  # newest token present

    def test_fewer_tokens_than_sink_no_eviction(self, tiny_config) -> None:
        """When seq_len < sink_tokens, nothing should be evicted."""
        cache = StreamingLLMCache(tiny_config, sink_tokens=4)
        sliding_idx = 0

        k = torch.randn(1, tiny_config.num_kv_heads, 2, 32)
        v = torch.randn(1, tiny_config.num_kv_heads, 2, 32)
        k_out, _ = cache.update(k, v, sliding_idx, torch.arange(2))

        assert k_out.shape[-2] == 2  # no eviction
        assert cache.cache_positions(sliding_idx).shape[0] == 2

    def test_cache_size_stays_at_window_after_many_decodes(self, tiny_config) -> None:
        """Cache size must never exceed window_size regardless of how many tokens are decoded."""
        cache = StreamingLLMCache(tiny_config, sink_tokens=4)
        sliding_idx = 0
        window = tiny_config.layer_params(sliding_idx).window_size  # 64

        # Decode 3× the window one token at a time
        for step in range(window * 3):
            k = torch.randn(1, tiny_config.num_kv_heads, 1, 32)
            v = torch.randn(1, tiny_config.num_kv_heads, 1, 32)
            k_out, _ = cache.update(k, v, sliding_idx, torch.tensor([step]))
            assert k_out.shape[-2] <= window, f"step {step}: cache exceeded window"

        pos = cache.cache_positions(sliding_idx)
        assert pos is not None
        assert pos.shape[0] == window
        # Sinks always at positions 0-3
        for i in range(4):
            assert pos[i].item() == i

    def test_sink_tokens_never_exceed_window_budget(self, tiny_config) -> None:
        sliding_idx = 0
        window = tiny_config.layer_params(sliding_idx).window_size  # 64
        cache = StreamingLLMCache(tiny_config, sink_tokens=window + 10)

        k = torch.randn(1, tiny_config.num_kv_heads, window + 20, 32)
        v = torch.randn(1, tiny_config.num_kv_heads, window + 20, 32)
        k_out, _ = cache.update(k, v, sliding_idx, torch.arange(window + 20))

        assert k_out.shape[-2] == window
        assert cache.last_evictions[sliding_idx] == 20


class TestH2OCache:
    def test_accumulates_scores_per_layer(self, tiny_config) -> None:
        cache = H2OCache(tiny_config)
        k = torch.randn(1, tiny_config.num_kv_heads, 4, 32)
        v = torch.randn(1, tiny_config.num_kv_heads, 4, 32)
        cache.update(k, v, layer_idx=0)

        attn = torch.softmax(torch.randn(1, tiny_config.num_heads, 1, 4), dim=-1)
        cache.update_scores(0, attn)

        assert 0 in cache.scores
        assert cache.scores[0].shape[0] == 4

    def test_retains_high_score_token_over_oldest(self, tiny_config) -> None:
        """Token with highest attention score survives even if it is the oldest."""
        cache = H2OCache(tiny_config)
        sliding_idx = 0
        window = tiny_config.layer_params(sliding_idx).window_size  # 64

        # Fill to window; mark first token distinctively
        k = torch.zeros(1, tiny_config.num_kv_heads, window, 32)
        v = torch.zeros(1, tiny_config.num_kv_heads, window, 32)
        k[0, :, 0, 0] = 999.0
        cache.update(k, v, sliding_idx)

        # Give token 0 (oldest) maximum attention score
        attn = torch.zeros(1, tiny_config.num_heads, 1, window)
        attn[0, :, 0, 0] = 1.0
        cache.update_scores(sliding_idx, attn)

        # Add new token — should evict a low-score token, NOT token 0
        k_new = torch.zeros(1, tiny_config.num_kv_heads, 1, 32)
        v_new = torch.zeros(1, tiny_config.num_kv_heads, 1, 32)
        k_out, _ = cache.update(k_new, v_new, sliding_idx)

        assert k_out.shape[-2] == window
        assert (k_out[0, 0, :, 0] == 999.0).any(), "high-score token 0 should be retained"

    def test_global_layers_unaffected(self, tiny_config) -> None:
        cache = H2OCache(tiny_config)
        global_idx = 3

        k = torch.randn(1, tiny_config.num_kv_heads, 100, 64)
        v = torch.randn(1, tiny_config.num_kv_heads, 100, 64)
        k_out, _ = cache.update(k, v, global_idx)

        assert k_out.shape[-2] == 100

    def test_reset_clears_scores(self, tiny_config) -> None:
        cache = H2OCache(tiny_config)
        k = torch.randn(1, tiny_config.num_kv_heads, 4, 32)
        v = torch.randn(1, tiny_config.num_kv_heads, 4, 32)
        cache.update(k, v, layer_idx=0)
        attn = torch.softmax(torch.randn(1, tiny_config.num_heads, 1, 4), dim=-1)
        cache.update_scores(0, attn)

        cache.reset()
        assert cache.scores == {}

    def test_falls_back_to_fifo_without_scores(self, tiny_config) -> None:
        """Without any score updates, H2O degrades to FIFO eviction."""
        cache = H2OCache(tiny_config)
        sliding_idx = 0
        window = tiny_config.layer_params(sliding_idx).window_size  # 64

        k = torch.zeros(1, tiny_config.num_kv_heads, window + 5, 32)
        for i in range(window + 5):
            k[0, :, i, 0] = float(i)
        v = torch.zeros_like(k)
        k_out, _ = cache.update(k, v, sliding_idx)

        assert k_out.shape[-2] == window
        # No scores → FIFO: keep last window tokens
        assert k_out[0, 0, 0, 0].item() == float(5)  # oldest kept is index 5

    def test_score_vector_length_matches_cache_after_eviction(self, tiny_config) -> None:
        """After eviction, scores[layer_idx] must have exactly window_size entries."""
        cache = H2OCache(tiny_config)
        sliding_idx = 0
        window = tiny_config.layer_params(sliding_idx).window_size  # 64

        k = torch.randn(1, tiny_config.num_kv_heads, window, 32)
        v = torch.randn(1, tiny_config.num_kv_heads, window, 32)
        cache.update(k, v, sliding_idx)

        attn = torch.softmax(torch.randn(1, tiny_config.num_heads, 1, window), dim=-1)
        cache.update_scores(sliding_idx, attn)

        # Add one more token — triggers eviction
        k2 = torch.randn(1, tiny_config.num_kv_heads, 1, 32)
        v2 = torch.randn(1, tiny_config.num_kv_heads, 1, 32)
        cache.update(k2, v2, sliding_idx)

        assert cache.scores[sliding_idx].shape[0] == window

    def test_multiple_score_updates_accumulate(self, tiny_config) -> None:
        """Scores must accumulate (add) across decode steps, not reset each time."""
        cache = H2OCache(tiny_config)
        sliding_idx = 0

        k = torch.randn(1, tiny_config.num_kv_heads, 4, 32)
        v = torch.randn(1, tiny_config.num_kv_heads, 4, 32)
        cache.update(k, v, sliding_idx)

        attn1 = torch.zeros(1, tiny_config.num_heads, 1, 4)
        attn1[0, :, 0, 0] = 1.0
        cache.update_scores(sliding_idx, attn1)
        score_after_first = cache.scores[sliding_idx][0].item()

        cache.update_scores(sliding_idx, attn1)
        score_after_second = cache.scores[sliding_idx][0].item()

        assert abs(score_after_second - 2 * score_after_first) < 1e-5

    def test_new_token_initialized_at_mean_score(self, tiny_config) -> None:
        """New tokens must be initialised at mean of existing scores."""
        cache = H2OCache(tiny_config)
        sliding_idx = 0
        window = tiny_config.layer_params(sliding_idx).window_size  # 64

        k = torch.randn(1, tiny_config.num_kv_heads, window, 32)
        v = torch.randn(1, tiny_config.num_kv_heads, window, 32)
        cache.update(k, v, sliding_idx)

        attn = torch.full((1, tiny_config.num_heads, 1, window), 1.0 / window)
        cache.update_scores(sliding_idx, attn)

        k2 = torch.randn(1, tiny_config.num_kv_heads, 1, 32)
        v2 = torch.randn(1, tiny_config.num_kv_heads, 1, 32)
        cache.update(k2, v2, sliding_idx)

        assert cache.seq_length(sliding_idx) == window
