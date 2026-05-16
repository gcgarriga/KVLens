"""Tests for Gemma4Config and GenerationConfig."""

import pytest

from kvlens.config import Gemma4Config, GenerationConfig, LayerParams, RoPEConfig


class TestGemma4Config:
    def test_e4b_defaults(self) -> None:
        cfg = Gemma4Config()
        assert cfg.num_layers == 42
        assert cfg.hidden_size == 2560
        assert cfg.num_heads == 8
        assert cfg.num_kv_heads == 2
        assert cfg.gqa_ratio == 4
        assert cfg.vocab_size == 262144

    def test_layer_types_auto_generated(self) -> None:
        cfg = Gemma4Config()
        assert len(cfg.layer_types) == 42
        # Global layers at indices 5, 11, 17, 23, 29, 35, 41
        global_indices = [i for i, t in enumerate(cfg.layer_types) if t == "global"]
        assert global_indices == [5, 11, 17, 23, 29, 35, 41]
        sliding_count = sum(1 for t in cfg.layer_types if t == "sliding")
        assert sliding_count == 35

    def test_layer_params_sliding(self) -> None:
        cfg = Gemma4Config()
        p = cfg.layer_params(0)
        assert p.head_dim == 256
        assert p.rope.theta == 10000.0
        assert p.rope.partial_factor == 1.0
        assert p.window_size == 512

    def test_layer_params_global(self) -> None:
        cfg = Gemma4Config()
        p = cfg.layer_params(5)
        assert p.head_dim == 512
        assert p.rope.theta == 1_000_000.0
        assert p.rope.partial_factor == 0.25
        assert p.window_size is None

    def test_layer_params_same_type_share_object(self) -> None:
        cfg = Gemma4Config()
        assert cfg.layer_params(0) is cfg.layer_params(1)  # both sliding
        assert cfg.layer_params(5) is cfg.layer_params(11)  # both global

    def test_tiny_config(self) -> None:
        cfg = Gemma4Config.tiny()
        assert cfg.num_layers == 4
        assert cfg.hidden_size == 128
        assert cfg.layer_types == ("sliding", "sliding", "sliding", "global")
        assert cfg.layer_params(0).head_dim == 32
        assert cfg.layer_params(3).head_dim == 64
        assert cfg.gqa_ratio == 2

    def test_invalid_kv_heads(self) -> None:
        with pytest.raises(ValueError, match="divisible"):
            Gemma4Config(num_heads=8, num_kv_heads=3)

    def test_invalid_layer_types_length(self) -> None:
        with pytest.raises(ValueError, match="layer_types length"):
            Gemma4Config(num_layers=4, layer_types=("sliding", "sliding"))

    def test_invalid_layer_type_value(self) -> None:
        with pytest.raises(ValueError, match="must be 'sliding' or 'global'"):
            Gemma4Config(num_layers=2, layer_types=("sliding", "invalid"))

    def test_frozen(self) -> None:
        cfg = Gemma4Config()
        with pytest.raises(AttributeError):
            cfg.num_layers = 10  # type: ignore[misc]


class TestGenerationConfig:
    def test_defaults(self) -> None:
        cfg = GenerationConfig()
        assert cfg.max_tokens == 128
        assert cfg.temperature == 0.0
        assert cfg.cache_strategy == "standard"

    def test_invalid_max_tokens(self) -> None:
        with pytest.raises(ValueError, match="max_tokens"):
            GenerationConfig(max_tokens=0)

    def test_invalid_temperature(self) -> None:
        with pytest.raises(ValueError, match="temperature"):
            GenerationConfig(temperature=-1.0)

    def test_invalid_top_p(self) -> None:
        with pytest.raises(ValueError, match="top_p"):
            GenerationConfig(top_p=0.0)
        with pytest.raises(ValueError, match="top_p"):
            GenerationConfig(top_p=1.5)

    def test_invalid_cache_strategy(self) -> None:
        with pytest.raises(ValueError, match="cache_strategy"):
            GenerationConfig(cache_strategy="unknown")

    def test_valid_strategies(self) -> None:
        for strategy in ("naive", "standard", "quantized"):
            cfg = GenerationConfig(cache_strategy=strategy)
            assert cfg.cache_strategy == strategy


class TestLayerParams:
    def test_sliding_bundle(self) -> None:
        p = LayerParams(head_dim=256, rope=RoPEConfig(), window_size=512)
        assert p.head_dim == 256
        assert p.window_size == 512

    def test_global_bundle(self) -> None:
        rope = RoPEConfig(theta=1e6, partial_factor=0.25)
        p = LayerParams(head_dim=512, rope=rope, window_size=None)
        assert p.head_dim == 512
        assert p.window_size is None


def test_gemma2_2b_config_constructs() -> None:
    config = Gemma4Config.gemma2_2b()
    assert config.num_layers == 26
    assert config.ple_dim == 0
    assert config.qk_norm is False
    assert config.max_position_embeddings == 8192
    assert config.final_logit_softcapping == 30.0
    assert config.num_kv_shared_layers == 0
    assert config.layer_types[0] == "sliding"
    assert config.layer_types[1] == "global"


class TestRoPEConfig:
    def test_defaults(self) -> None:
        cfg = RoPEConfig()
        assert cfg.theta == 10000.0
        assert cfg.partial_factor == 1.0

    def test_custom(self) -> None:
        cfg = RoPEConfig(theta=1_000_000.0, partial_factor=0.25)
        assert cfg.theta == 1_000_000.0
        assert cfg.partial_factor == 0.25
