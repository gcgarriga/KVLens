"""Smoke tests for paper figure generation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from kvlens.experiments.plot import (
    _bootstrap_ci,
    _latency_budget_values,
    _mean_decode_ms,
    plot_category_breakdown,
    plot_eviction_pressure,
    plot_kl_delta,
    plot_kl_pareto,
    plot_latency_pareto,
    plot_layer_type_breakdown,
    plot_pareto,
    plot_perplexity_delta,
    plot_prompt_length_effect,
    plot_survival_heatmap,
)
from kvlens.experiments.runner import RunResult


@pytest.fixture
def sample_results() -> list[RunResult]:
    results = []
    for strategy in ("standard", "streaming", "h2o"):
        for budget in (32, 64):
            for prompt_idx in (0, 1):
                results.append(
                    RunResult(
                        strategy=strategy,
                        window_budget=budget,
                        prompt_idx=prompt_idx,
                        kl_divergence=0.1 if strategy != "standard" else 0.0,
                        perplexity=5.0 if strategy == "standard" else 6.0,
                        memory_bytes_sliding=budget * 32 * 35,
                        memory_bytes_global=0,
                        token_survival_rates={i: 0.9 for i in range(4)},
                        prompt_tokens=80 + prompt_idx * 40,
                        generated_tokens=10,
                        effective_tokens=90 + prompt_idx * 40,
                        evicted_tokens_sliding=budget if strategy != "standard" else 0,
                        evicted_tokens_global=0,
                        eviction_rate_sliding=0.5 if strategy != "standard" else 0.0,
                        eviction_rate_global=0.0,
                    )
                )
    return results


def test_plot_pareto_produces_file(sample_results) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "pareto.png"
        plot_pareto(sample_results, out)
        assert out.exists()
        assert out.stat().st_size > 0


def test_plot_latency_pareto_produces_file_with_timings(sample_results) -> None:
    # Sample fixture has decode_ms_per_token=0; populate latencies for this test
    # so the plot has a real curve rather than the empty-data fallback.
    for i, r in enumerate(sample_results):
        r.decode_ms_per_token = 1.0 + 0.1 * i
        r.prefill_ms = 10.0
        r.total_decode_ms = r.decode_ms_per_token * 10
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "latency_pareto.png"
        plot_latency_pareto(sample_results, out)
        assert out.exists()
        assert out.stat().st_size > 0


def test_plot_latency_pareto_handles_missing_timings(sample_results) -> None:
    # Legacy per-layer budget results have decode_ms_per_token=0; plot must still render.
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "latency_pareto.png"
        plot_latency_pareto(sample_results, out)
        assert out.exists()
        assert out.stat().st_size > 0


def test_latency_pareto_groups_iso_memory_results_by_memory_budget() -> None:
    results = [
        RunResult(
            strategy="h2o",
            memory_budget_mb=1,
            prompt_idx=0,
            memory_bytes_sliding=1024,
            decode_ms_per_token=1.0,
        ),
        RunResult(
            strategy="h2o",
            memory_budget_mb=2,
            prompt_idx=0,
            memory_bytes_sliding=2048,
            decode_ms_per_token=3.0,
        ),
    ]

    budgets, iso_memory = _latency_budget_values(results)

    assert iso_memory is True
    assert budgets == [1, 2]
    assert _mean_decode_ms(results, "h2o", 1, iso_memory) == 1.0
    assert _mean_decode_ms(results, "h2o", 2, iso_memory) == 3.0


@pytest.mark.parametrize(
    ("plot_func", "filename"),
    [
        (plot_perplexity_delta, "perplexity_delta.png"),
        (plot_kl_delta, "kl_delta.png"),
        (plot_eviction_pressure, "eviction_pressure.png"),
        (plot_prompt_length_effect, "prompt_length_effect.png"),
    ],
)
def test_diagnostic_plots_produce_files(sample_results, plot_func, filename) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / filename
        plot_func(sample_results, out)
        assert out.exists()
        assert out.stat().st_size > 0


def test_plot_survival_heatmap_produces_file(sample_results) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "heatmap.png"
        plot_survival_heatmap(sample_results, out, num_layers=4)
        assert out.exists()
        assert out.stat().st_size > 0


def test_plot_layer_type_breakdown_produces_file(sample_results) -> None:
    from kvlens.config import Gemma4Config

    config = Gemma4Config.tiny()
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "breakdown.png"
        plot_layer_type_breakdown(
            sample_results, out, layer_types=config.layer_types, target_budget=64
        )
        assert out.exists()
        assert out.stat().st_size > 0


class TestBootstrapCI:
    def test_returns_nan_triple_for_empty_input(self) -> None:
        mean, lo, hi = _bootstrap_ci([])
        assert all(v != v for v in (mean, lo, hi))  # NaN check

    def test_skips_nan_values(self) -> None:
        mean, lo, hi = _bootstrap_ci([1.0, float("nan"), 2.0, float("nan"), 3.0])
        assert abs(mean - 2.0) < 1e-9
        assert lo <= mean <= hi

    def test_single_finite_value_collapses_ci_to_point(self) -> None:
        mean, lo, hi = _bootstrap_ci([5.0])
        assert mean == lo == hi == 5.0

    def test_seeded_rng_makes_output_deterministic(self) -> None:
        values = [1.2, 2.4, 3.6, 4.8, 6.0, 7.2, 8.4]
        first = _bootstrap_ci(values)
        second = _bootstrap_ci(values)
        assert first == second

    def test_ci_brackets_the_mean(self) -> None:
        # Bounded interval should contain the mean for any seed.
        mean, lo, hi = _bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0])
        assert lo <= mean <= hi


def test_plot_category_breakdown_produces_file_with_error_bars() -> None:
    results = []
    for strategy in ("standard", "h2o"):
        for cat in ("retrieval", "creative"):
            for prompt_idx in range(4):
                results.append(
                    RunResult(
                        strategy=strategy,
                        window_budget=16,
                        prompt_idx=prompt_idx,
                        kl_divergence=0.0 if strategy == "standard" else 0.2,
                        perplexity=2.0 + 0.1 * prompt_idx,
                        memory_bytes_sliding=1024,
                        memory_bytes_global=2048,
                        category=cat,
                        prompt_tokens=100,
                        generated_tokens=10,
                        effective_tokens=110,
                    )
                )
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "category_breakdown.png"
        plot_category_breakdown(results, out, target_budget=16)
        assert out.exists()
        assert out.stat().st_size > 0


def test_plot_kl_pareto_produces_file_with_ci_band(sample_results) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "kl_pareto.png"
        plot_kl_pareto(sample_results, out)
        assert out.exists()
        assert out.stat().st_size > 0
