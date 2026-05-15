"""Tests for explicit attention."""

from __future__ import annotations

import math

import torch
from torch import nn

from kvlens.attention import Attention
from kvlens.config import Gemma4Config, LayerParams, RoPEConfig


def make_attention_config(
    *,
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    window_size: int | None,
) -> Gemma4Config:
    params = LayerParams(
        head_dim=head_dim,
        rope=RoPEConfig(theta=1e30, partial_factor=1.0),
        window_size=window_size,
    )
    return Gemma4Config(
        num_layers=1,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        vocab_size=32,
        max_position_embeddings=32,
        layer_types=("global" if window_size is None else "sliding",),
        sliding=params,
        global_=params,
        ple_dim=4,
        num_kv_shared_layers=0,
    )


class TestAttention:
    def test_output_matches_hand_computed_values(self) -> None:
        class IdentityRoPE(nn.Module):
            def forward(
                self,
                x: torch.Tensor,
                positions: torch.Tensor | None = None,
            ) -> torch.Tensor:
                return x

        config = make_attention_config(
            hidden_size=2,
            num_heads=1,
            num_kv_heads=1,
            head_dim=2,
            window_size=None,
        )
        attention = Attention(config, layer_idx=0)
        attention.rope = IdentityRoPE()

        with torch.no_grad():
            eye = torch.eye(2)
            attention.q_proj.weight.copy_(eye)
            attention.k_proj.weight.copy_(eye)
            attention.v_proj.weight.copy_(eye)
            attention.o_proj.weight.copy_(eye)
            attention.q_norm.weight.fill_(0.0)
            attention.k_norm.weight.fill_(0.0)

        hidden_states = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        output = attention(hidden_states, positions=torch.tensor([0, 1]))

        norm = math.sqrt(0.5)
        token0 = torch.tensor([1.0 / norm, 0.0])
        token1 = torch.tensor([0.0, 1.0 / norm])
        score = (token1 @ token0).item()
        self_score = (token1 @ token1).item()
        weight0 = math.exp(score) / (math.exp(score) + math.exp(self_score))
        weight1 = math.exp(self_score) / (math.exp(score) + math.exp(self_score))

        expected = torch.stack(
            [
                token0,
                (weight0 * token0) + (weight1 * token1),
            ],
            dim=0,
        ).unsqueeze(0)

        torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-6)

    def test_gqa_broadcast_repeats_kv_heads(self) -> None:
        config = make_attention_config(
            hidden_size=4,
            num_heads=4,
            num_kv_heads=2,
            head_dim=2,
            window_size=None,
        )
        attention = Attention(config, layer_idx=0)
        kv = torch.arange(1, 17, dtype=torch.float32).view(1, 2, 4, 2)

        expanded = attention._expand_kv(kv)

        assert expanded.shape == (1, 4, 4, 2)
        torch.testing.assert_close(expanded[:, 0], kv[:, 0])
        torch.testing.assert_close(expanded[:, 1], kv[:, 0])
        torch.testing.assert_close(expanded[:, 2], kv[:, 1])
        torch.testing.assert_close(expanded[:, 3], kv[:, 1])

    def test_sliding_window_mask_blocks_old_tokens(self) -> None:
        config = make_attention_config(
            hidden_size=4,
            num_heads=1,
            num_kv_heads=1,
            head_dim=2,
            window_size=2,
        )
        attention = Attention(config, layer_idx=0)

        mask = attention._build_attention_mask(torch.tensor([0, 1, 2, 3]), key_len=4)

        assert mask.shape == (1, 1, 4, 4)
        assert not mask[0, 0, 3, 0].item()
        assert not mask[0, 0, 3, 1].item()
        assert mask[0, 0, 3, 2].item()
        assert mask[0, 0, 3, 3].item()

    def test_explicit_cached_positions_are_not_native_window_masked(self) -> None:
        config = make_attention_config(
            hidden_size=4,
            num_heads=1,
            num_kv_heads=1,
            head_dim=2,
            window_size=2,
        )
        attention = Attention(config, layer_idx=0)

        mask = attention._build_attention_mask(
            torch.tensor([5]),
            key_len=4,
            cached_positions=torch.tensor([0, 3, 4, 5]),
        )

        assert mask.shape == (1, 1, 1, 4)
        assert mask[0, 0, 0, 0].item()
        assert mask[0, 0, 0, 1].item()
        assert mask[0, 0, 0, 2].item()
        assert mask[0, 0, 0, 3].item()

    def test_global_attention_sees_full_history(self) -> None:
        config = make_attention_config(
            hidden_size=4,
            num_heads=1,
            num_kv_heads=1,
            head_dim=2,
            window_size=None,
        )
        attention = Attention(config, layer_idx=0)

        mask = attention._build_attention_mask(torch.tensor([0, 1, 2, 3]), key_len=4)

        assert mask[0, 0, 3, 0].item()
        assert mask[0, 0, 3, 1].item()
        assert mask[0, 0, 3, 2].item()
        assert mask[0, 0, 3, 3].item()

    def test_layer_types_use_different_head_dims(self, tiny_config: Gemma4Config) -> None:
        sliding_attention = Attention(tiny_config, layer_idx=0)
        global_attention = Attention(tiny_config, layer_idx=3)

        assert sliding_attention.head_dim == 32
        assert global_attention.head_dim == 64
        assert sliding_attention.q_proj.weight.shape == (128, 128)
        assert global_attention.q_proj.weight.shape == (256, 128)


class PositionTrackingCache:
    """Fake cache that records positions passed to update()."""

    def __init__(self) -> None:
        self.received_positions: dict[int, torch.Tensor] = {}
        self.storage: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int,
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if positions is not None:
            self.received_positions[layer_idx] = positions.clone()
        self.storage[layer_idx] = (key, value)
        return key, value

    def get(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        return self.storage.get(layer_idx)

    def reset(self) -> None:
        self.storage.clear()

    def seq_length(self, layer_idx: int) -> int:
        return 0

    def key_offset(self, layer_idx: int) -> int:
        return 0

    def memory_bytes(self, layer_idx: int) -> int:
        return 0


def test_attention_passes_positions_to_cache(tiny_config: Gemma4Config) -> None:
    attn = Attention(tiny_config, layer_idx=0)
    fake_cache = PositionTrackingCache()
    hidden = torch.randn(1, 3, tiny_config.hidden_size)
    positions = torch.tensor([5, 6, 7])
    attn(hidden, positions=positions, cache=fake_cache)
    assert 0 in fake_cache.received_positions
    torch.testing.assert_close(fake_cache.received_positions[0], positions)


class ScoreTrackingCache:
    """Fake cache that records update_scores calls."""

    def __init__(self) -> None:
        self.score_calls: list[tuple[int, torch.Tensor]] = []
        self.storage: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int,
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.storage[layer_idx] = (key, value)
        return key, value

    def update_scores(self, layer_idx: int, attention_weights: torch.Tensor) -> None:
        self.score_calls.append((layer_idx, attention_weights.clone()))

    def get(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        return self.storage.get(layer_idx)

    def reset(self) -> None:
        self.storage.clear()

    def seq_length(self, layer_idx: int) -> int:
        return 0

    def key_offset(self, layer_idx: int) -> int:
        return 0

    def memory_bytes(self, layer_idx: int) -> int:
        return 0


def test_attention_calls_update_scores_on_decode(tiny_config: Gemma4Config) -> None:
    attn = Attention(tiny_config, layer_idx=0)
    fake_cache = ScoreTrackingCache()
    hidden = torch.randn(1, 1, tiny_config.hidden_size)  # q_len=1 (decode)
    positions = torch.tensor([10])
    attn(hidden, positions=positions, cache=fake_cache)
    assert len(fake_cache.score_calls) == 1
    assert fake_cache.score_calls[0][0] == 0


def test_attention_skips_update_scores_on_prefill(tiny_config: Gemma4Config) -> None:
    attn = Attention(tiny_config, layer_idx=0)
    fake_cache = ScoreTrackingCache()
    hidden = torch.randn(1, 5, tiny_config.hidden_size)  # q_len=5 (prefill)
    positions = torch.arange(5)
    attn(hidden, positions=positions, cache=fake_cache)
    assert len(fake_cache.score_calls) == 0  # only called during decode


class ExplicitPositionCache:
    """Fake cache that exposes explicit (non-contiguous) token positions.

    Tracks positions as tokens are added via update(), mirroring how a real
    StreamingLLM cache would maintain its own position index.
    """

    def __init__(self, positions: torch.Tensor) -> None:
        self._positions = positions
        self._kv: tuple[torch.Tensor, torch.Tensor] | None = None

    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int,
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._kv is None:
            self._kv = (key, value)
        else:
            self._kv = (
                torch.cat([self._kv[0], key], dim=-2),
                torch.cat([self._kv[1], value], dim=-2),
            )
        # Track newly added positions
        if positions is not None:
            self._positions = torch.cat([self._positions, positions])
        return self._kv

    def cache_positions(self, layer_idx: int) -> torch.Tensor:
        return self._positions

    def get(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        return self._kv

    def reset(self) -> None:
        self._kv = None

    def seq_length(self, layer_idx: int) -> int:
        return self._kv[0].shape[-2] if self._kv else 0

    def key_offset(self, layer_idx: int) -> int:
        return 0

    def memory_bytes(self, layer_idx: int) -> int:
        return 0


def test_attention_uses_cache_positions_for_mask(tiny_config: Gemma4Config) -> None:
    """When cache provides explicit positions, attention should not raise."""
    attn = Attention(tiny_config, layer_idx=0)

    sink_positions = torch.tensor([0, 1, 2, 3, 100, 101, 102, 103])
    fake_cache = ExplicitPositionCache(sink_positions)

    k = torch.randn(1, tiny_config.num_kv_heads, 8, 32)
    v = torch.randn(1, tiny_config.num_kv_heads, 8, 32)
    fake_cache.update(k, v, layer_idx=0)

    hidden = torch.randn(1, 1, tiny_config.hidden_size)
    positions = torch.tensor([110])
    output = attn(hidden, positions=positions, cache=fake_cache)
    assert output.shape == (1, 1, tiny_config.hidden_size)


class PrefillScoreCache:
    """Fake cache that records accumulate_prefill_scores calls."""

    def __init__(self) -> None:
        self.prefill_calls: list[tuple[int, torch.Tensor]] = []
        self.decode_calls: list[int] = []
        self.storage: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int,
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.storage[layer_idx] = (key, value)
        return key, value

    def accumulate_prefill_scores(self, layer_idx: int, attention_weights: torch.Tensor) -> None:
        self.prefill_calls.append((layer_idx, attention_weights.clone()))

    def update_scores(self, layer_idx: int, attention_weights: torch.Tensor) -> None:
        self.decode_calls.append(layer_idx)

    def get(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        return self.storage.get(layer_idx)

    def reset(self) -> None:
        self.storage.clear()

    def seq_length(self, layer_idx: int) -> int:
        return 0

    def key_offset(self, layer_idx: int) -> int:
        return 0

    def memory_bytes(self, layer_idx: int) -> int:
        return 0


def test_attention_calls_accumulate_prefill_scores_on_prefill(tiny_config: Gemma4Config) -> None:
    attn = Attention(tiny_config, layer_idx=0)
    fake_cache = PrefillScoreCache()
    hidden = torch.randn(1, 5, tiny_config.hidden_size)  # q_len=5 (prefill)
    positions = torch.arange(5)
    attn(hidden, positions=positions, cache=fake_cache)
    assert len(fake_cache.prefill_calls) == 1
    assert fake_cache.prefill_calls[0][0] == 0


def test_attention_skips_accumulate_prefill_scores_on_decode(tiny_config: Gemma4Config) -> None:
    attn = Attention(tiny_config, layer_idx=0)
    fake_cache = PrefillScoreCache()
    hidden = torch.randn(1, 1, tiny_config.hidden_size)  # q_len=1 (decode)
    positions = torch.tensor([10])
    attn(hidden, positions=positions, cache=fake_cache)
    assert len(fake_cache.prefill_calls) == 0  # only called during prefill
    assert len(fake_cache.decode_calls) == 1  # update_scores still called
