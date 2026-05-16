"""Tests for the ``all_layers_*`` canonical-paper baselines.

Each ``all_layers_*`` strategy must evict from global layers when the window
budget is small, in contrast to the corresponding hybrid-adapted variant which
preserves global context.
"""

from __future__ import annotations

import pytest
import torch

from kvlens.cache import create_cache
from kvlens.config import Gemma4Config

KVPair = tuple[torch.Tensor, torch.Tensor]


@pytest.fixture
def tiny_config() -> Gemma4Config:
    return Gemma4Config.tiny()


GLOBAL_IDX = 3
SLIDING_IDX = 0
BUDGET = 8
PROMPT_LEN = 24
DECODE_STEPS = 3


def _make_kv(config: Gemma4Config, layer_idx: int, length: int) -> KVPair:
    head_dim = config.layer_params(layer_idx).head_dim
    k = torch.randn(1, config.num_kv_heads, length, head_dim)
    v = torch.randn(1, config.num_kv_heads, length, head_dim)
    return k, v


def _prefill_then_decode(
    cache: object, config: Gemma4Config, layer_idx: int, prompt_len: int, steps: int
) -> None:
    # SnapKV only triggers eviction on the first decode step, so all canonical
    # baselines need at least one (prefill, decode) sequence to compare fairly.
    k_p, v_p = _make_kv(config, layer_idx, prompt_len)
    cache.update(k_p, v_p, layer_idx, positions=torch.arange(prompt_len))  # type: ignore[attr-defined]
    for step in range(steps):
        pos = prompt_len + step
        k_d, v_d = _make_kv(config, layer_idx, 1)
        cache.update(k_d, v_d, layer_idx, positions=torch.tensor([pos]))  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "canonical, hybrid",
    [
        ("all_layers_h2o", "h2o"),
        ("all_layers_snapkv", "snapkv"),
        ("all_layers_streaming", "streaming"),
        ("all_layers_pyramidkv", "pyramidkv"),
    ],
)
def test_canonical_baseline_evicts_globals(
    tiny_config: Gemma4Config, canonical: str, hybrid: str
) -> None:
    """``all_layers_*`` evicts from globals; the hybrid-adapted variant does not."""
    assert tiny_config.layer_types[GLOBAL_IDX] == "global"

    canonical_cache = create_cache(canonical, tiny_config, window_budget=BUDGET)
    hybrid_cache = create_cache(hybrid, tiny_config, window_budget=BUDGET)

    _prefill_then_decode(canonical_cache, tiny_config, GLOBAL_IDX, PROMPT_LEN, DECODE_STEPS)
    _prefill_then_decode(hybrid_cache, tiny_config, GLOBAL_IDX, PROMPT_LEN, DECODE_STEPS)

    assert hybrid_cache.key_offset(GLOBAL_IDX) == 0, (  # type: ignore[attr-defined]
        f"{hybrid} must keep full context on global layers (no eviction)"
    )
    assert canonical_cache.key_offset(GLOBAL_IDX) > 0, (  # type: ignore[attr-defined]
        f"{canonical} must evict from global layers (canonical-paper behaviour)"
    )
    assert canonical_cache.seq_length(GLOBAL_IDX) <= BUDGET + 1, (  # type: ignore[attr-defined]
        f"{canonical} global cache must respect the window budget"
    )


@pytest.mark.parametrize(
    "canonical",
    ["all_layers_h2o", "all_layers_snapkv", "all_layers_streaming", "all_layers_pyramidkv"],
)
def test_canonical_baseline_evicts_sliding(tiny_config: Gemma4Config, canonical: str) -> None:
    """Sanity: canonical baselines must also evict from sliding layers."""
    cache = create_cache(canonical, tiny_config, window_budget=BUDGET)
    _prefill_then_decode(cache, tiny_config, SLIDING_IDX, PROMPT_LEN, DECODE_STEPS)
    assert cache.key_offset(SLIDING_IDX) > 0  # type: ignore[attr-defined]
    assert cache.seq_length(SLIDING_IDX) <= BUDGET + 1  # type: ignore[attr-defined]


def test_layertype_h2o_factory_removed(tiny_config: Gemma4Config) -> None:
    """``layertype_h2o`` is no longer a valid factory strategy (redundant with ``h2o``)."""
    with pytest.raises(ValueError, match="layertype_h2o"):
        create_cache("layertype_h2o", tiny_config)


def test_pyramidkv_apply_to_all_layers_assigns_global_budget(tiny_config: Gemma4Config) -> None:
    """When apply_to_all_layers=True, globals get a finite budget instead of None."""
    cache = create_cache("all_layers_pyramidkv", tiny_config, window_budget=BUDGET)
    assert cache._budgets[GLOBAL_IDX] is not None  # type: ignore[attr-defined]
    assert cache._budgets[SLIDING_IDX] is not None  # type: ignore[attr-defined]


def test_default_pyramidkv_keeps_globals_unbounded(tiny_config: Gemma4Config) -> None:
    """The hybrid-adapted ``pyramidkv`` keeps globals unbounded (window_size=None)."""
    cache = create_cache("pyramidkv", tiny_config, window_budget=BUDGET)
    assert cache._budgets[GLOBAL_IDX] is None  # type: ignore[attr-defined]
