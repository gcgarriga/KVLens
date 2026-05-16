from pathlib import Path

from kvlens.experiments.runner import RunResult, save_results
from kvlens.experiments.validate import (
    STRESS_SMOKE_BUDGETS,
    STRESS_SMOKE_FIGURES,
    STRESS_SMOKE_STRATEGIES,
    format_stress_validation_report,
    validate_stress_smoke,
)


def _write_figures(figures_dir: Path, names: tuple[str, ...] = STRESS_SMOKE_FIGURES) -> None:
    figures_dir.mkdir()
    for name in names:
        (figures_dir / name).write_bytes(b"png")


def _sample_results() -> list[RunResult]:
    results = []
    for prompt_idx in range(2):
        for strategy in STRESS_SMOKE_STRATEGIES:
            for budget in STRESS_SMOKE_BUDGETS:
                is_standard = strategy == "standard"
                is_all_layers = strategy == "all_layers_h2o"
                results.append(
                    RunResult(
                        strategy=strategy,
                        window_budget=budget,
                        prompt_idx=prompt_idx,
                        kl_divergence=0.0 if is_standard else 0.01,
                        perplexity=5.0 if is_standard else 5.5,
                        memory_bytes_sliding=budget * 1024,
                        memory_bytes_global=budget * 128,
                        token_survival_rates={0: 0.5},
                        category="retrieval",
                        prompt_tokens=128,
                        generated_tokens=16,
                        effective_tokens=144,
                        evicted_tokens_sliding=32 if not is_standard else 0,
                        evicted_tokens_global=16 if is_all_layers else 0,
                        eviction_rate_sliding=0.25 if not is_standard else 0.0,
                        eviction_rate_global=0.25 if is_all_layers else 0.0,
                        cache_lengths_by_layer={0: budget},
                        evictions_by_layer={0: 32 if not is_standard else 0},
                    )
                )
    return results


def test_validate_stress_smoke_accepts_complete_artifacts(tmp_path: Path) -> None:
    results_path = tmp_path / "stress_smoke_results.json"
    figures_dir = tmp_path / "figures"
    save_results(_sample_results(), results_path)
    _write_figures(figures_dir)

    report = validate_stress_smoke(results_path, figures_dir)

    assert report.ok
    assert report.run_count == 56
    assert report.nonstandard_positive_kl == 48
    assert report.max_sliding_eviction_rate == 0.25
    assert report.max_global_eviction_rate == 0.25
    assert "status=ok" in format_stress_validation_report(report)


def test_validate_stress_smoke_rejects_missing_combo(tmp_path: Path) -> None:
    results_path = tmp_path / "stress_smoke_results.json"
    figures_dir = tmp_path / "figures"
    results = _sample_results()
    save_results(results[:-1], results_path)
    _write_figures(figures_dir)

    report = validate_stress_smoke(results_path, figures_dir)

    assert not report.ok
    assert any("missing 1 prompt-strategy-budget runs" in error for error in report.errors)


def test_validate_stress_smoke_rejects_missing_figure(tmp_path: Path) -> None:
    results_path = tmp_path / "stress_smoke_results.json"
    figures_dir = tmp_path / "figures"
    save_results(_sample_results(), results_path)
    _write_figures(figures_dir, names=STRESS_SMOKE_FIGURES[:-1])

    report = validate_stress_smoke(results_path, figures_dir)

    assert not report.ok
    assert any("prompt_length_effect.png" in error for error in report.errors)


def test_validate_stress_smoke_requires_global_h2o_contrast(tmp_path: Path) -> None:
    results_path = tmp_path / "stress_smoke_results.json"
    figures_dir = tmp_path / "figures"
    results = _sample_results()
    for result in results:
        if result.strategy == "all_layers_h2o":
            result.eviction_rate_global = 0.0
    save_results(results, results_path)
    _write_figures(figures_dir)

    report = validate_stress_smoke(results_path, figures_dir)

    assert not report.ok
    assert any(
        "all_layers_h2o did not evict global-layer tokens" in error for error in report.errors
    )
