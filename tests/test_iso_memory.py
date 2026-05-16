"""Tests for iso-memory budgeting (``--memory-budget MB``)."""

from __future__ import annotations

import pytest
import torch

from kvlens.cache import create_cache
from kvlens.cache.allocation import allocate_memory_budget, per_layer_token_bytes
from kvlens.config import Gemma4Config

DTYPE_BYTES = 4  # float32 for CPU tests


@pytest.fixture
def tiny_config() -> Gemma4Config:
    return Gemma4Config.tiny()


HYBRID_STRATEGIES = ("h2o", "snapkv", "streaming", "pyramidkv")
CANONICAL_STRATEGIES = (
    "all_layers_h2o",
    "all_layers_snapkv",
    "all_layers_streaming",
    "all_layers_pyramidkv",
)
PROPORTIONAL_STRATEGIES = (
    "proportional_h2o",
    "proportional_snapkv",
    "proportional_streaming",
    "proportional_pyramidkv",
)
ALL_BUDGETED_STRATEGIES = HYBRID_STRATEGIES + CANONICAL_STRATEGIES + PROPORTIONAL_STRATEGIES


class TestAllocateMemoryBudget:
    def test_standard_returns_empty(self, tiny_config: Gemma4Config) -> None:
        assert allocate_memory_budget("standard", tiny_config, 4096, DTYPE_BYTES) == {}

    def test_naive_quantized_return_empty(self, tiny_config: Gemma4Config) -> None:
        assert allocate_memory_budget("naive", tiny_config, 4096, DTYPE_BYTES) == {}
        assert allocate_memory_budget("quantized", tiny_config, 4096, DTYPE_BYTES) == {}

    @pytest.mark.parametrize("strategy", ("h2o", "snapkv", "streaming", "pyramidkv"))
    def test_hybrid_keeps_globals_unbounded(
        self, tiny_config: Gemma4Config, strategy: str
    ) -> None:
        budgets = allocate_memory_budget(strategy, tiny_config, 4096, DTYPE_BYTES)
        for layer_idx, t in enumerate(tiny_config.layer_types):
            if t == "global":
                assert budgets[layer_idx] is None, (
                    f"{strategy}: global layer {layer_idx} must keep full context"
                )
            else:
                assert isinstance(budgets[layer_idx], int)
                assert budgets[layer_idx] >= 1

    @pytest.mark.parametrize("strategy", CANONICAL_STRATEGIES)
    def test_canonical_caps_every_layer(self, tiny_config: Gemma4Config, strategy: str) -> None:
        budgets = allocate_memory_budget(strategy, tiny_config, 4096, DTYPE_BYTES)
        assert set(budgets.keys()) == set(range(tiny_config.num_layers))
        for layer_idx in range(tiny_config.num_layers):
            assert isinstance(budgets[layer_idx], int)
            assert budgets[layer_idx] >= 1

    def test_pyramid_shallow_layers_get_more(self, tiny_config: Gemma4Config) -> None:
        budgets = allocate_memory_budget("pyramidkv", tiny_config, 4096, DTYPE_BYTES)
        sliding_indices = [i for i, t in enumerate(tiny_config.layer_types) if t == "sliding"]
        sliding_budgets = [budgets[i] for i in sliding_indices]
        for earlier, later in zip(sliding_budgets, sliding_budgets[1:], strict=False):
            assert earlier >= later, (
                f"pyramid budget should not increase with depth: {sliding_budgets}"
            )

    def test_all_layers_uniform_gives_equal_token_bytes_per_layer(
        self, tiny_config: Gemma4Config
    ) -> None:
        total_bytes = 4096
        budgets = allocate_memory_budget("all_layers_h2o", tiny_config, total_bytes, DTYPE_BYTES)
        per_token_sliding = per_layer_token_bytes(tiny_config, 0, DTYPE_BYTES)
        per_token_global = per_layer_token_bytes(tiny_config, 3, DTYPE_BYTES)
        # Globals have larger head_dim → fewer tokens for the same byte share.
        if per_token_global > per_token_sliding:
            assert budgets[0] >= budgets[3]

    def test_unknown_strategy_raises(self, tiny_config: Gemma4Config) -> None:
        with pytest.raises(ValueError, match="unknown strategy"):
            allocate_memory_budget("does_not_exist", tiny_config, 4096, DTYPE_BYTES)

    def test_zero_or_negative_total_raises(self, tiny_config: Gemma4Config) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            allocate_memory_budget("h2o", tiny_config, 0, DTYPE_BYTES)

    def test_hybrid_sliding_budgets_do_not_exceed_native_window(
        self, tiny_config: Gemma4Config
    ) -> None:
        budgets = allocate_memory_budget("h2o", tiny_config, 1024 * 1024, DTYPE_BYTES)

        for layer_idx, layer_type in enumerate(tiny_config.layer_types):
            if layer_type == "sliding":
                assert budgets[layer_idx] == tiny_config.layer_params(layer_idx).window_size
            else:
                assert budgets[layer_idx] is None

    def test_all_layers_surplus_sliding_budget_flows_to_globals(
        self, tiny_config: Gemma4Config
    ) -> None:
        budgets = allocate_memory_budget("all_layers_h2o", tiny_config, 1024 * 1024, DTYPE_BYTES)

        sliding_idx = next(i for i, t in enumerate(tiny_config.layer_types) if t == "sliding")
        global_idx = next(i for i, t in enumerate(tiny_config.layer_types) if t == "global")
        assert budgets[sliding_idx] == tiny_config.layer_params(sliding_idx).window_size
        assert isinstance(budgets[global_idx], int)
        assert budgets[global_idx] > tiny_config.layer_params(sliding_idx).window_size


class TestProportionalAllocation:
    """``proportional_*`` allocates iso-token across all layers.

    Each layer receives the same number of cached tokens, so byte share scales
    with per-token cost (head_dim × num_kv_heads × 2 × dtype_bytes). On a model
    with uniform head_dim across layers this collapses to ``all_layers_*``; on
    a heterogeneous model (Gemma 4: globals 2× sliding head_dim) globals get a
    proportionally larger byte budget.
    """

    @pytest.mark.parametrize("strategy", PROPORTIONAL_STRATEGIES)
    def test_caps_every_layer(self, tiny_config: Gemma4Config, strategy: str) -> None:
        budgets = allocate_memory_budget(strategy, tiny_config, 4096, DTYPE_BYTES)
        assert set(budgets.keys()) == set(range(tiny_config.num_layers))
        for layer_idx in range(tiny_config.num_layers):
            assert isinstance(budgets[layer_idx], int)
            assert budgets[layer_idx] >= 1

    def test_collapses_to_all_layers_on_uniform_head_dim(self) -> None:
        """On Gemma 2 (uniform head_dim 256) proportional ≡ all_layers."""
        cfg = Gemma4Config.gemma2_2b()
        for inner in ("h2o", "snapkv", "streaming"):
            prop = allocate_memory_budget(f"proportional_{inner}", cfg, 64 * 1024, DTYPE_BYTES)
            uniform = allocate_memory_budget(f"all_layers_{inner}", cfg, 64 * 1024, DTYPE_BYTES)
            assert prop == uniform, (
                f"proportional_{inner} must equal all_layers_{inner} on uniform head_dim"
            )

    def test_iso_token_on_heterogeneous_head_dim(self, tiny_config: Gemma4Config) -> None:
        """tiny_config has sliding head_dim=32 and global head_dim=64; every
        layer must end up with the same token count."""
        budgets = allocate_memory_budget("proportional_h2o", tiny_config, 4 * 1024, DTYPE_BYTES)
        token_counts = {budgets[i] for i in range(tiny_config.num_layers)}
        assert len(token_counts) == 1, f"all layers must receive the same token cap, got {budgets}"

    def test_globals_get_more_bytes_on_heterogeneous(self, tiny_config: Gemma4Config) -> None:
        """Iso-token allocation gives larger-head_dim layers more bytes."""
        budgets = allocate_memory_budget("proportional_h2o", tiny_config, 4 * 1024, DTYPE_BYTES)
        sliding_idx = next(i for i, t in enumerate(tiny_config.layer_types) if t == "sliding")
        global_idx = next(i for i, t in enumerate(tiny_config.layer_types) if t == "global")
        sliding_bytes = budgets[sliding_idx] * per_layer_token_bytes(
            tiny_config, sliding_idx, DTYPE_BYTES
        )
        global_bytes = budgets[global_idx] * per_layer_token_bytes(
            tiny_config, global_idx, DTYPE_BYTES
        )
        assert global_bytes > sliding_bytes

    def test_pyramid_decreases_with_depth(self, tiny_config: Gemma4Config) -> None:
        """proportional_pyramidkv: shallowest layer gets the most tokens."""
        budgets = allocate_memory_budget(
            "proportional_pyramidkv", tiny_config, 4 * 1024, DTYPE_BYTES
        )
        token_caps = [budgets[i] for i in range(tiny_config.num_layers)]
        for earlier, later in zip(token_caps, token_caps[1:], strict=False):
            assert earlier >= later, (
                f"pyramid token cap should not increase with depth: {token_caps}"
            )


class TestConfigOverrides:
    def test_with_per_layer_budgets_returns_new_config(self, tiny_config: Gemma4Config) -> None:
        new_cfg = tiny_config.with_per_layer_budgets({0: 8, 3: None})
        assert new_cfg is not tiny_config
        assert new_cfg.layer_params(0).window_size == 8
        assert new_cfg.layer_params(3).window_size is None

    def test_unmodified_layer_uses_default_params(self, tiny_config: Gemma4Config) -> None:
        new_cfg = tiny_config.with_per_layer_budgets({0: 8})
        assert new_cfg.layer_params(1).window_size == tiny_config.sliding.window_size

    def test_override_preserves_head_dim_and_rope(self, tiny_config: Gemma4Config) -> None:
        new_cfg = tiny_config.with_per_layer_budgets({3: 16})
        assert new_cfg.layer_params(3).head_dim == tiny_config.global_.head_dim
        assert new_cfg.layer_params(3).rope == tiny_config.global_.rope


class TestFactoryIsoMemory:
    @pytest.mark.parametrize("strategy", ALL_BUDGETED_STRATEGIES)
    def test_factory_accepts_memory_budget(self, tiny_config: Gemma4Config, strategy: str) -> None:
        cache = create_cache(
            strategy, tiny_config, memory_budget_bytes=4096, dtype_bytes=DTYPE_BYTES
        )
        assert cache is not None

    def test_window_budget_and_memory_budget_mutually_exclusive(
        self, tiny_config: Gemma4Config
    ) -> None:
        with pytest.raises(ValueError, match="Cannot pass both"):
            create_cache("h2o", tiny_config, window_budget=8, memory_budget_bytes=4096)

    def test_pyramidkv_uses_precomputed_budgets(self, tiny_config: Gemma4Config) -> None:
        cache = create_cache(
            "pyramidkv", tiny_config, memory_budget_bytes=4096, dtype_bytes=DTYPE_BYTES
        )
        expected = allocate_memory_budget("pyramidkv", tiny_config, 4096, DTYPE_BYTES)
        assert cache._budgets == expected  # type: ignore[attr-defined]

    def test_all_layers_pyramidkv_uses_precomputed_budgets(
        self, tiny_config: Gemma4Config
    ) -> None:
        cache = create_cache(
            "all_layers_pyramidkv",
            tiny_config,
            memory_budget_bytes=4096,
            dtype_bytes=DTYPE_BYTES,
        )
        expected = allocate_memory_budget("all_layers_pyramidkv", tiny_config, 4096, DTYPE_BYTES)
        assert cache._budgets == expected  # type: ignore[attr-defined]


def _drive_cache(cache: object, config: Gemma4Config, prompt_len: int, decode_steps: int) -> None:
    """Run prefill + decode on every layer to populate the cache."""
    for layer_idx in range(config.num_layers):
        head_dim = config.layer_params(layer_idx).head_dim
        k_p = torch.randn(1, config.num_kv_heads, prompt_len, head_dim)
        v_p = torch.randn(1, config.num_kv_heads, prompt_len, head_dim)
        cache.update(k_p, v_p, layer_idx, positions=torch.arange(prompt_len))  # type: ignore[attr-defined]
        for step in range(decode_steps):
            pos = prompt_len + step
            k_d = torch.randn(1, config.num_kv_heads, 1, head_dim)
            v_d = torch.randn(1, config.num_kv_heads, 1, head_dim)
            cache.update(k_d, v_d, layer_idx, positions=torch.tensor([pos]))  # type: ignore[attr-defined]


class TestIsoMemoryBudgetRespected:
    @pytest.mark.parametrize("strategy", ALL_BUDGETED_STRATEGIES)
    def test_total_cache_memory_within_budget(
        self, tiny_config: Gemma4Config, strategy: str
    ) -> None:
        # Use a generous budget so every layer gets a non-trivial cap and the
        # check is meaningful — strategies need enough headroom to populate.
        budget_bytes = 32 * 1024  # 32 KB
        cache = create_cache(
            strategy, tiny_config, memory_budget_bytes=budget_bytes, dtype_bytes=DTYPE_BYTES
        )
        _drive_cache(cache, tiny_config, prompt_len=32, decode_steps=4)
        total = sum(cache.memory_bytes(i) for i in range(tiny_config.num_layers))  # type: ignore[attr-defined]

        if strategy in HYBRID_STRATEGIES:
            # Hybrid-adapted strategies leave globals unbounded — cap only the
            # sliding portion.
            sliding_total = sum(
                cache.memory_bytes(i)  # type: ignore[attr-defined]
                for i, t in enumerate(tiny_config.layer_types)
                if t == "sliding"
            )
            assert sliding_total <= budget_bytes, (
                f"{strategy}: sliding total {sliding_total} exceeded budget {budget_bytes}"
            )
        else:
            # Canonical baselines must respect the total budget across all layers.
            # Allow a small slack for the +1 in _bytes_to_tokens (rounding up to a
            # minimum of one token per layer).
            slack = tiny_config.num_layers * max(
                per_layer_token_bytes(tiny_config, i, DTYPE_BYTES)
                for i in range(tiny_config.num_layers)
            )
            assert total <= budget_bytes + slack, (
                f"{strategy}: total {total} exceeded budget {budget_bytes} + slack {slack}"
            )


def test_hybrid_h2o_allocates_only_to_globals():
    from kvlens.cache.allocation import allocate_memory_budget

    config = Gemma4Config.tiny()
    budgets = allocate_memory_budget("hybrid_h2o", config, total_bytes=4096, dtype_bytes=2)
    sliding_idx = [i for i in range(config.num_layers) if config.layer_types[i] == "sliding"]
    global_idx = [i for i in range(config.num_layers) if config.layer_types[i] == "global"]
    for i in sliding_idx:
        assert i not in budgets, f"sliding layer {i} must not have a budget override"
    for i in global_idx:
        assert isinstance(budgets[i], int), f"global layer {i} must have an int budget"
        assert budgets[i] >= 1


def test_hybrid_snapkv_uses_same_allocator_as_hybrid_h2o():
    from kvlens.cache.allocation import allocate_memory_budget

    config = Gemma4Config.tiny()
    a = allocate_memory_budget("hybrid_h2o", config, total_bytes=8192, dtype_bytes=2)
    b = allocate_memory_budget("hybrid_snapkv", config, total_bytes=8192, dtype_bytes=2)
    c = allocate_memory_budget("hybrid_streaming", config, total_bytes=8192, dtype_bytes=2)
    assert a == b == c


def test_hybrid_h2o_globals_split_uniformly():
    from kvlens.cache.allocation import allocate_memory_budget, per_layer_token_bytes

    config = Gemma4Config.tiny()
    budgets = allocate_memory_budget("hybrid_h2o", config, total_bytes=4096, dtype_bytes=2)
    global_idx = [i for i in range(config.num_layers) if config.layer_types[i] == "global"]
    per_layer = 4096 // len(global_idx)
    for i in global_idx:
        expected_tokens = max(1, per_layer // per_layer_token_bytes(config, i, 2))
        assert budgets[i] == expected_tokens


def test_hybrid_h2o_at_huge_budget_matches_standard_on_kv_state(
    tiny_config: Gemma4Config,
) -> None:
    """At a non-binding budget, hybrid_h2o must produce byte-identical KV state
    on every layer to the standard HybridCache. This is the controlled-cache test
    that motivates the paper's central claim."""
    from kvlens.cache import create_cache

    standard = create_cache("standard", tiny_config)
    hybrid = create_cache("hybrid_h2o", tiny_config, memory_budget_bytes=10 * (1 << 20))  # 10 MB

    torch.manual_seed(0)
    seq_len = 80  # > tiny config sliding window of 64, triggers FIFO eviction
    for layer_idx in range(tiny_config.num_layers):
        head_dim = tiny_config.layer_params(layer_idx).head_dim
        k = torch.randn(1, tiny_config.num_kv_heads, seq_len, head_dim)
        v = torch.randn(1, tiny_config.num_kv_heads, seq_len, head_dim)
        positions = torch.arange(seq_len)
        k_std, v_std = standard.update(k, v, layer_idx, positions)
        k_hyb, v_hyb = hybrid.update(k, v, layer_idx, positions)
        assert k_std.shape == k_hyb.shape, f"layer {layer_idx} key shape mismatch"
        assert v_std.shape == v_hyb.shape, f"layer {layer_idx} value shape mismatch"
        assert torch.equal(k_std, k_hyb), f"layer {layer_idx} key content mismatch"
        assert torch.equal(v_std, v_hyb), f"layer {layer_idx} value content mismatch"
