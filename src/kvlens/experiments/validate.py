"""Validation helpers for experiment artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from kvlens.experiments.runner import RunResult, load_results

STRESS_SMOKE_STRATEGIES = (
    "standard",
    "streaming",
    "h2o",
    "snapkv",
    "pyramidkv",
    "layertype_h2o",
    "all_layers_h2o",
)
STRESS_SMOKE_BUDGETS = (8, 16, 32, 64)
STRESS_SMOKE_FIGURES = (
    "pareto.png",
    "kl_pareto.png",
    "survival_heatmap.png",
    "layer_type_breakdown.png",
    "category_breakdown.png",
    "perplexity_delta.png",
    "kl_delta.png",
    "eviction_pressure.png",
    "prompt_length_effect.png",
)


@dataclass(frozen=True)
class StressValidationReport:
    results_path: Path
    figures_dir: Path
    run_count: int
    prompt_count: int
    strategies: tuple[str, ...]
    budgets: tuple[int, ...]
    max_sliding_eviction_rate: float
    max_global_eviction_rate: float
    nonstandard_positive_kl: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _missing_combinations(
    results: list[RunResult],
    prompt_indices: set[int],
    strategies: tuple[str, ...],
    budgets: tuple[int, ...],
) -> list[tuple[int, str, int]]:
    seen = {(r.prompt_idx, r.strategy, r.window_budget) for r in results}
    missing = []
    for prompt_idx in sorted(prompt_indices):
        for strategy in strategies:
            for budget in budgets:
                combo = (prompt_idx, strategy, budget)
                if combo not in seen:
                    missing.append(combo)
    return missing


def validate_stress_smoke(
    results_path: Path,
    figures_dir: Path,
    expected_strategies: tuple[str, ...] = STRESS_SMOKE_STRATEGIES,
    expected_budgets: tuple[int, ...] = STRESS_SMOKE_BUDGETS,
    required_figures: tuple[str, ...] = STRESS_SMOKE_FIGURES,
) -> StressValidationReport:
    """Validate legacy stress-smoke artifacts."""
    errors: list[str] = []
    warnings: list[str] = []

    if not results_path.exists():
        return StressValidationReport(
            results_path=results_path,
            figures_dir=figures_dir,
            run_count=0,
            prompt_count=0,
            strategies=(),
            budgets=(),
            max_sliding_eviction_rate=0.0,
            max_global_eviction_rate=0.0,
            nonstandard_positive_kl=0,
            errors=[f"missing results file: {results_path}"],
        )

    results = load_results(results_path)
    prompt_indices = {r.prompt_idx for r in results}
    strategies = tuple(sorted({r.strategy for r in results}))
    budgets = tuple(sorted({r.window_budget for r in results}))

    missing_strategies = sorted(set(expected_strategies) - set(strategies))
    if missing_strategies:
        errors.append(f"missing strategies: {', '.join(missing_strategies)}")

    missing_budgets = sorted(set(expected_budgets) - set(budgets))
    if missing_budgets:
        errors.append(f"missing budgets: {', '.join(str(b) for b in missing_budgets)}")

    missing_combos = _missing_combinations(
        results, prompt_indices, expected_strategies, expected_budgets
    )
    if missing_combos:
        sample = ", ".join(f"prompt={p}/strategy={s}/budget={b}" for p, s, b in missing_combos[:5])
        suffix = "..." if len(missing_combos) > 5 else ""
        errors.append(
            f"missing {len(missing_combos)} prompt-strategy-budget runs: {sample}{suffix}"
        )

    standard_runs = [r for r in results if r.strategy == "standard"]
    if not standard_runs:
        errors.append("missing standard baseline runs")
    elif any(r.kl_divergence != 0.0 for r in standard_runs):
        errors.append("standard baseline KL must be exactly 0.0")

    if any(r.perplexity <= 0.0 for r in results):
        errors.append("all runs must have positive perplexity")

    nonstandard_positive_kl = sum(
        1 for r in results if r.strategy != "standard" and r.kl_divergence > 0.0
    )
    if nonstandard_positive_kl == 0:
        errors.append("no non-standard run has positive KL divergence")

    max_sliding_eviction_rate = max((r.eviction_rate_sliding for r in results), default=0.0)
    max_global_eviction_rate = max((r.eviction_rate_global for r in results), default=0.0)
    if max_sliding_eviction_rate <= 0.0:
        errors.append("sliding layers never evicted tokens")

    all_layers_global = [r.eviction_rate_global for r in results if r.strategy == "all_layers_h2o"]
    if not all_layers_global or max(all_layers_global) <= 0.0:
        errors.append("all_layers_h2o did not evict global-layer tokens")

    layertype_global = [r.eviction_rate_global for r in results if r.strategy == "layertype_h2o"]
    if any(rate > 0.0 for rate in layertype_global):
        errors.append("layertype_h2o unexpectedly evicted global-layer tokens")

    missing_figures = []
    empty_figures = []
    for figure in required_figures:
        path = figures_dir / figure
        if not path.exists():
            missing_figures.append(figure)
        elif path.stat().st_size == 0:
            empty_figures.append(figure)
    if missing_figures:
        errors.append(f"missing figures: {', '.join(missing_figures)}")
    if empty_figures:
        errors.append(f"empty figures: {', '.join(empty_figures)}")

    if prompt_indices and len(prompt_indices) < 12:
        warnings.append(f"stress-smoke used only {len(prompt_indices)} prompts")

    return StressValidationReport(
        results_path=results_path,
        figures_dir=figures_dir,
        run_count=len(results),
        prompt_count=len(prompt_indices),
        strategies=strategies,
        budgets=budgets,
        max_sliding_eviction_rate=max_sliding_eviction_rate,
        max_global_eviction_rate=max_global_eviction_rate,
        nonstandard_positive_kl=nonstandard_positive_kl,
        errors=errors,
        warnings=warnings,
    )


def format_stress_validation_report(report: StressValidationReport) -> str:
    """Render a validation report for CLI/script output."""
    lines = [
        f"results={report.results_path}",
        f"figures={report.figures_dir}",
        f"runs={report.run_count}",
        f"prompts={report.prompt_count}",
        f"strategies={','.join(report.strategies)}",
        f"budgets={','.join(str(b) for b in report.budgets)}",
        f"max_sliding_eviction_rate={report.max_sliding_eviction_rate:.4f}",
        f"max_global_eviction_rate={report.max_global_eviction_rate:.4f}",
        f"nonstandard_positive_kl={report.nonstandard_positive_kl}",
    ]
    for warning in report.warnings:
        lines.append(f"WARNING: {warning}")
    for error in report.errors:
        lines.append(f"ERROR: {error}")
    lines.append("status=ok" if report.ok else "status=failed")
    return "\n".join(lines)
