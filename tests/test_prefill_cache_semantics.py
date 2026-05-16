"""Regression tests for prefill logits and decode-cache materialization."""

from __future__ import annotations

import torch

from kvlens.cache import create_cache
from kvlens.config import Gemma4Config, LayerParams, RoPEConfig
from kvlens.generation import decode_step, prefill
from kvlens.model import GemmaModel


def _windowed_config(*, shared_layers: int = 0) -> Gemma4Config:
    return Gemma4Config(
        num_layers=6,
        hidden_size=128,
        intermediate_size=256,
        num_heads=4,
        num_kv_heads=2,
        vocab_size=512,
        max_position_embeddings=256,
        layer_types=("sliding", "sliding", "global", "sliding", "sliding", "global"),
        sliding=LayerParams(head_dim=32, rope=RoPEConfig(), window_size=16),
        global_=LayerParams(
            head_dim=32,
            rope=RoPEConfig(theta=1_000_000.0),
            window_size=None,
        ),
        ple_dim=0,
        num_kv_shared_layers=shared_layers,
    )


def test_prefill_with_standard_cache_matches_no_cache_above_sliding_window() -> None:
    torch.manual_seed(123)
    config = _windowed_config(shared_layers=0)
    model = GemmaModel(config).eval()
    input_ids = (torch.arange(40).view(1, 40) % config.vocab_size).long()
    cache = create_cache("standard", config)

    with torch.inference_mode():
        expected = prefill(model, input_ids, cache=None)
        actual = prefill(model, input_ids, cache=cache)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_prefill_with_standard_cache_matches_no_cache_with_kv_sharing() -> None:
    torch.manual_seed(123)
    config = _windowed_config(shared_layers=2)
    model = GemmaModel(config).eval()
    input_ids = (torch.arange(40).view(1, 40) % config.vocab_size).long()
    cache = create_cache("standard", config)

    with torch.inference_mode():
        expected = prefill(model, input_ids, cache=None)
        actual = prefill(model, input_ids, cache=cache)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_decode_after_prefill_matches_no_cache_with_kv_sharing() -> None:
    torch.manual_seed(123)
    config = _windowed_config(shared_layers=2)
    model = GemmaModel(config).eval()
    input_ids = (torch.arange(40).view(1, 40) % config.vocab_size).long()
    next_token = torch.tensor([[41]])
    cache = create_cache("standard", config)

    with torch.inference_mode():
        prefill(model, input_ids, cache=cache)
        actual = decode_step(model, next_token, position=40, cache=cache)
        expected = model(torch.cat([input_ids, next_token], dim=1))[:, -1:, :]

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_multi_token_continuation_prefill_uses_existing_cache_context() -> None:
    torch.manual_seed(123)
    config = _windowed_config(shared_layers=0)
    model = GemmaModel(config).eval()
    prefix_ids = (torch.arange(40).view(1, 40) % config.vocab_size).long()
    continuation_ids = torch.tensor([[41, 42, 43]])
    cache = create_cache("standard", config)

    with torch.inference_mode():
        prefill(model, prefix_ids, cache=cache)
        actual = prefill(
            model,
            continuation_ids,
            cache=cache,
            position_offset=prefix_ids.shape[1],
        )
        expected = model(torch.cat([prefix_ids, continuation_ids], dim=1))[:, -3:, :]

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
