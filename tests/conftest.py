"""Shared test fixtures for KVLens."""

import pytest

from kvlens.config import Gemma4Config, LayerParams, RoPEConfig


@pytest.fixture
def tiny_config() -> Gemma4Config:
    """Tiny config for CPU tests: 4 layers (3 sliding + 1 global), small dims."""
    return Gemma4Config.tiny()


@pytest.fixture
def tiny_gemma2_config() -> Gemma4Config:
    """Gemma 2-style tiny config: ple_dim=0, softcap=0, alternating layers, no KV sharing."""
    return Gemma4Config(
        num_layers=4,
        hidden_size=128,
        intermediate_size=512,
        num_heads=4,
        num_kv_heads=2,
        vocab_size=256,
        max_position_embeddings=512,
        layer_types=("sliding", "global", "sliding", "global"),
        sliding=LayerParams(head_dim=32, rope=RoPEConfig(), window_size=64),
        global_=LayerParams(head_dim=32, rope=RoPEConfig(), window_size=None),
        ple_dim=0,
        qk_norm=False,
        final_logit_softcapping=30.0,
        num_kv_shared_layers=0,
    )
