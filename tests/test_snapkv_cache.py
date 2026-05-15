"""Tests for SnapKVCache."""

from __future__ import annotations

import pytest
import torch

from kvlens.cache.snapkv import SnapKVCache
from kvlens.config import Gemma4Config


@pytest.fixture
def tiny_config() -> Gemma4Config:
    return Gemma4Config.tiny()


class TestSnapKVCache:
    def test_no_eviction_during_prefill(self, tiny_config) -> None:
        """During prefill, all tokens must be stored — selection happens at decode."""
        cache = SnapKVCache(tiny_config, observation_window=4)
        sliding_idx = 0
        k = torch.randn(1, tiny_config.num_kv_heads, 20, 32)
        v = torch.randn(1, tiny_config.num_kv_heads, 20, 32)
        k_out, _ = cache.update(k, v, sliding_idx, torch.arange(20))
        assert k_out.shape[-2] == 20  # no eviction yet

    def test_snap_on_first_decode_step(self, tiny_config) -> None:
        """On the first decode token, cache is trimmed to window_size by score."""
        cache = SnapKVCache(tiny_config, observation_window=4)
        sliding_idx = 0
        window = tiny_config.layer_params(sliding_idx).window_size  # 64

        # Prefill with window + 20 tokens
        k = torch.zeros(1, tiny_config.num_kv_heads, window + 20, 32)
        v = torch.zeros(1, tiny_config.num_kv_heads, window + 20, 32)
        # Mark first 4 tokens distinctively
        for i in range(4):
            k[0, :, i, 0] = float(i + 100)
        positions = torch.arange(window + 20)
        cache.update(k, v, sliding_idx, positions)

        # Provide prefill scores: tokens 0-3 have maximum attention
        attn = torch.zeros(1, tiny_config.num_kv_heads, window + 20, window + 20)
        attn[0, :, -4:, :4] = 1.0  # last 4 queries attend fully to first 4 tokens
        cache.accumulate_prefill_scores(sliding_idx, attn)

        # Verify snap state: all 4 high-score tokens kept before decode eviction
        cache._snap(sliding_idx)
        snapped = cache.storage[sliding_idx][0]
        snapped_vals = snapped[0, 0, :, 0]
        assert all((snapped_vals == float(i + 100)).any() for i in range(4))
        assert snapped.shape[-2] == window

        # First decode step — appends 1 token → 65 > 64 → FIFO drops token at position 0.
        # Snap sorts kept indices by original position, so token 0 (val=100) is evicted first.
        k2 = torch.randn(1, tiny_config.num_kv_heads, 1, 32)
        v2 = torch.randn(1, tiny_config.num_kv_heads, 1, 32)
        k_out, _ = cache.update(k2, v2, sliding_idx, torch.tensor([window + 20]))

        assert k_out.shape[-2] == window
        # Tokens 1-3 (vals 101-103) survive; token 0 (val=100) was FIFO-evicted
        retained_vals = k_out[0, 0, :, 0]
        assert all((retained_vals == float(i + 100)).any() for i in range(1, 4))

    def test_snap_happens_only_once(self, tiny_config) -> None:
        """After the first decode step, subsequent decode steps do not re-snap."""
        cache = SnapKVCache(tiny_config, observation_window=4)
        sliding_idx = 0
        window = tiny_config.layer_params(sliding_idx).window_size  # 64

        k = torch.randn(1, tiny_config.num_kv_heads, window + 10, 32)
        v = torch.randn(1, tiny_config.num_kv_heads, window + 10, 32)
        cache.update(k, v, sliding_idx, torch.arange(window + 10))

        attn = torch.softmax(torch.randn(1, tiny_config.num_kv_heads, 5, window + 10), dim=-1)
        cache.accumulate_prefill_scores(sliding_idx, attn)

        # First decode
        k2 = torch.randn(1, tiny_config.num_kv_heads, 1, 32)
        v2 = torch.randn(1, tiny_config.num_kv_heads, 1, 32)
        k_out1, _ = cache.update(k2, v2, sliding_idx, torch.tensor([window + 10]))
        size_after_first = k_out1.shape[-2]

        # Second decode — cache grows by 1 (no second snap)
        k3 = torch.randn(1, tiny_config.num_kv_heads, 1, 32)
        v3 = torch.randn(1, tiny_config.num_kv_heads, 1, 32)
        k_out2, _ = cache.update(k3, v3, sliding_idx, torch.tensor([window + 11]))
        assert k_out2.shape[-2] == min(size_after_first + 1, window)

    def test_last_evictions_preserves_snap_transition(self, tiny_config) -> None:
        cache = SnapKVCache(tiny_config, observation_window=4)
        sliding_idx = 0
        window = tiny_config.layer_params(sliding_idx).window_size  # 64

        k = torch.randn(1, tiny_config.num_kv_heads, window + 10, 32)
        v = torch.randn(1, tiny_config.num_kv_heads, window + 10, 32)
        cache.update(k, v, sliding_idx, torch.arange(window + 10))

        k2 = torch.randn(1, tiny_config.num_kv_heads, 1, 32)
        v2 = torch.randn(1, tiny_config.num_kv_heads, 1, 32)
        cache.update(k2, v2, sliding_idx, torch.tensor([window + 10]))

        assert cache.last_evictions[sliding_idx] == 11
        assert cache.key_offset(sliding_idx) == 11

    def test_global_layers_unaffected(self, tiny_config) -> None:
        cache = SnapKVCache(tiny_config, observation_window=4)
        global_idx = 3
        k = torch.randn(1, tiny_config.num_kv_heads, 100, 64)
        v = torch.randn(1, tiny_config.num_kv_heads, 100, 64)
        cache.update(k, v, global_idx, torch.arange(100))
        # First decode
        k2 = torch.randn(1, tiny_config.num_kv_heads, 1, 64)
        v2 = torch.randn(1, tiny_config.num_kv_heads, 1, 64)
        k_out, _ = cache.update(k2, v2, global_idx, torch.tensor([100]))
        assert k_out.shape[-2] == 101  # no eviction on global layers

    def test_reset_clears_prefill_scores(self, tiny_config) -> None:
        cache = SnapKVCache(tiny_config, observation_window=4)
        sliding_idx = 0
        k = torch.randn(1, tiny_config.num_kv_heads, 4, 32)
        v = torch.randn(1, tiny_config.num_kv_heads, 4, 32)
        cache.update(k, v, sliding_idx, torch.arange(4))
        attn = torch.softmax(torch.randn(1, tiny_config.num_kv_heads, 4, 4), dim=-1)
        cache.accumulate_prefill_scores(sliding_idx, attn)
        cache.reset()
        assert cache._prefill_scores == {}
        assert cache._snapped == set()

    def test_no_scores_falls_back_to_fifo(self, tiny_config) -> None:
        """Without prefill score accumulation, SnapKV degrades to FIFO on snap."""
        cache = SnapKVCache(tiny_config, observation_window=4)
        sliding_idx = 0
        window = tiny_config.layer_params(sliding_idx).window_size  # 64

        k = torch.zeros(1, tiny_config.num_kv_heads, window + 10, 32)
        for i in range(window + 10):
            k[0, :, i, 0] = float(i)
        v = torch.zeros_like(k)
        cache.update(k, v, sliding_idx, torch.arange(window + 10))
        # No accumulate_prefill_scores call

        k2 = torch.randn(1, tiny_config.num_kv_heads, 1, 32)
        v2 = torch.randn(1, tiny_config.num_kv_heads, 1, 32)
        k_out, _ = cache.update(k2, v2, sliding_idx, torch.tensor([window + 10]))
        assert k_out.shape[-2] == window
        # FIFO: keep most recent window tokens
        assert k_out[0, 0, 0, 0].item() == float(10 + 1)
