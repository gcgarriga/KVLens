"""Tests for the batch experiment runner."""

from __future__ import annotations

import tempfile
from pathlib import Path

from kvlens.config import Gemma4Config
from kvlens.experiments.runner import (
    ExperimentConfig,
    RunResult,
    load_results,
    run_experiment,
    save_results,
)
from kvlens.model import GemmaModel


class FakeTokenizer:
    eos_id = 0
    eot_id = 0

    def encode(self, text: str) -> list[int]:
        return [1, 2, 3]

    def encode_chat_turn(self, text: str, role: str, is_first_turn: bool) -> list[int]:
        return [1, 2, 3]

    def decode(self, ids: list[int]) -> str:
        return "decoded"


def test_experiment_config_requires_one_budget_mode() -> None:
    import pytest

    with pytest.raises(ValueError, match="exactly one"):
        ExperimentConfig(prompts=["x"], strategies=["standard"])
    with pytest.raises(ValueError, match="exactly one"):
        ExperimentConfig(
            prompts=["x"], strategies=["standard"], window_budgets=(32,), memory_budgets_mb=(1,)
        )


def test_run_experiment_iso_memory_mode_populates_memory_budget_mb() -> None:
    config = Gemma4Config.tiny()
    model = GemmaModel(config)
    tokenizer = FakeTokenizer()

    exp_config = ExperimentConfig(
        prompts=["hello world"],
        strategies=["standard", "h2o", "all_layers_h2o", "all_layers_pyramidkv"],
        memory_budgets_mb=(1,),
        max_tokens=2,
        seed=0,
    )
    results = run_experiment(model, tokenizer, exp_config)

    assert len(results) == 4
    for r in results:
        assert r.memory_budget_mb == 1
        assert r.window_budget == 0


def test_run_experiment_returns_results_for_all_combinations() -> None:
    config = Gemma4Config.tiny()
    model = GemmaModel(config)
    tokenizer = FakeTokenizer()

    exp_config = ExperimentConfig(
        prompts=["hello world", "the quick brown fox"],
        strategies=["standard", "naive"],
        window_budgets=[32],
        max_tokens=3,
        seed=42,
    )
    results = run_experiment(model, tokenizer, exp_config)

    assert len(results) == 4  # 2 prompts × 2 strategies × 1 budget
    for r in results:
        assert isinstance(r, RunResult)
        assert r.strategy in ("standard", "naive")
        assert r.window_budget == 32
        assert r.prompt_idx in (0, 1)
        assert isinstance(r.kl_divergence, float)
        assert isinstance(r.perplexity, float)
        assert isinstance(r.memory_bytes_sliding, int)
        assert isinstance(r.memory_bytes_global, int)
        assert r.prompt_tokens == 3
        assert r.generated_tokens >= 0
        assert r.effective_tokens >= r.prompt_tokens
        assert 0.0 <= r.eviction_rate_sliding <= 1.0
        assert 0.0 <= r.eviction_rate_global <= 1.0
        assert isinstance(r.cache_lengths_by_layer, dict)
        assert isinstance(r.evictions_by_layer, dict)


def test_run_experiment_applies_prompt_index_offset() -> None:
    config = Gemma4Config.tiny()
    model = GemmaModel(config)
    tokenizer = FakeTokenizer()

    exp_config = ExperimentConfig(
        prompts=["chunk prompt a", "chunk prompt b"],
        strategies=["standard"],
        window_budgets=[32],
        max_tokens=2,
        seed=42,
        prompt_index_offset=10,
    )
    results = run_experiment(model, tokenizer, exp_config)

    assert [r.prompt_idx for r in results] == [10, 11]


def test_save_and_load_results_roundtrip() -> None:
    results = [
        RunResult(
            strategy="standard",
            window_budget=64,
            prompt_idx=0,
            kl_divergence=0.05,
            perplexity=3.2,
            memory_bytes_sliding=8192,
            memory_bytes_global=16384,
            token_survival_rates={0: 0.9, 1: 0.85},
            prompt_tokens=80,
            generated_tokens=12,
            effective_tokens=92,
            evicted_tokens_sliding=100,
            evicted_tokens_global=0,
            eviction_rate_sliding=0.5,
            eviction_rate_global=0.0,
            cache_lengths_by_layer={0: 64},
            evictions_by_layer={0: 64},
        )
    ]
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = Path(f.name)

    save_results(results, path)
    loaded = load_results(path)

    assert len(loaded) == 1
    assert loaded[0].strategy == "standard"
    assert abs(loaded[0].perplexity - 3.2) < 1e-6
    assert loaded[0].token_survival_rates == {0: 0.9, 1: 0.85}
    assert loaded[0].prompt_tokens == 80
    assert loaded[0].eviction_rate_sliding == 0.5
    assert loaded[0].cache_lengths_by_layer == {0: 64}


def test_load_results_supports_old_schema() -> None:
    old_json = """[
      {
        "strategy": "standard",
        "window_budget": 64,
        "prompt_idx": 0,
        "kl_divergence": 0.0,
        "perplexity": 3.2,
        "memory_bytes_sliding": 8192,
        "memory_bytes_global": 16384
      }
    ]"""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = Path(f.name)
    path.write_text(old_json)

    loaded = load_results(path)

    assert len(loaded) == 1
    assert loaded[0].category == "unknown"
    assert loaded[0].token_survival_rates == {}
    assert loaded[0].prompt_tokens == 0
    assert loaded[0].eviction_rate_sliding == 0.0


def test_run_experiment_records_decode_latency() -> None:
    config = Gemma4Config.tiny()
    model = GemmaModel(config)
    tokenizer = FakeTokenizer()

    exp_config = ExperimentConfig(
        prompts=["hello"],
        strategies=["standard", "h2o"],
        window_budgets=[32],
        max_tokens=4,
        seed=0,
    )
    results = run_experiment(model, tokenizer, exp_config)
    for r in results:
        assert r.prefill_ms > 0.0, f"{r.strategy}: prefill_ms must be populated"
        assert r.total_decode_ms >= 0.0
        # Equality only if decode_count was 0 (very short generation); otherwise
        # decode latency should be derived as average ms per token.
        if r.total_decode_ms > 0:
            assert r.decode_ms_per_token > 0


def test_load_results_defaults_latency_for_old_schema() -> None:
    old_json = """[
      {
        "strategy": "h2o",
        "window_budget": 32,
        "prompt_idx": 0,
        "kl_divergence": 0.1,
        "perplexity": 5.0,
        "memory_bytes_sliding": 1024,
        "memory_bytes_global": 0
      }
    ]"""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = Path(f.name)
    path.write_text(old_json)

    loaded = load_results(path)
    assert loaded[0].prefill_ms == 0.0
    assert loaded[0].total_decode_ms == 0.0
    assert loaded[0].decode_ms_per_token == 0.0


def test_standard_strategy_kl_divergence_is_zero() -> None:
    config = Gemma4Config.tiny()
    model = GemmaModel(config)
    tokenizer = FakeTokenizer()

    exp_config = ExperimentConfig(
        prompts=["hello"],
        strategies=["standard"],
        window_budgets=[32],
        max_tokens=3,
        seed=42,
    )
    results = run_experiment(model, tokenizer, exp_config)
    assert len(results) == 1
    assert results[0].kl_divergence == 0.0


def test_all_new_strategies_complete_without_error() -> None:
    config = Gemma4Config.tiny()
    model = GemmaModel(config)
    tokenizer = FakeTokenizer()

    exp_config = ExperimentConfig(
        prompts=["test prompt"],
        strategies=[
            "streaming",
            "h2o",
            "snapkv",
            "pyramidkv",
            "all_layers_h2o",
            "all_layers_snapkv",
            "all_layers_streaming",
            "all_layers_pyramidkv",
        ],
        window_budgets=[32],
        max_tokens=3,
        seed=0,
    )
    results = run_experiment(model, tokenizer, exp_config)
    assert len(results) == 8
    strategies_seen = {r.strategy for r in results}
    assert strategies_seen == {
        "streaming",
        "h2o",
        "snapkv",
        "pyramidkv",
        "all_layers_h2o",
        "all_layers_snapkv",
        "all_layers_streaming",
        "all_layers_pyramidkv",
    }
    for r in results:
        assert isinstance(r.kl_divergence, float)
        assert isinstance(r.perplexity, float)
        assert r.memory_bytes_sliding >= 0
        assert r.memory_bytes_global >= 0
        assert r.effective_tokens >= r.prompt_tokens
