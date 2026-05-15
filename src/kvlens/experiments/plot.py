"""Study figure generation from experiment results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from kvlens.experiments.runner import RunResult

STRATEGY_COLORS = {
    "standard": "#6b7280",
    "streaming": "#60a5fa",
    "h2o": "#f59e0b",
    "snapkv": "#818cf8",
    "pyramidkv": "#fb923c",
    "quantized": "#a78bfa",
    "naive": "#f87171",
    "all_layers_h2o": "#14b8a6",
    "all_layers_snapkv": "#0d9488",
    "all_layers_streaming": "#0891b2",
    "all_layers_pyramidkv": "#c2410c",
    # Retained for loading historical stress-smoke results from earlier experiments.
    "layertype_h2o": "#34d399",
}


CI_BOOTSTRAP_SEED = 42
CI_BOOTSTRAP_SAMPLES = 1000
CI_ALPHA = 0.05


def _bootstrap_ci(
    values: list[float],
    n: int = CI_BOOTSTRAP_SAMPLES,
    alpha: float = CI_ALPHA,
    seed: int = CI_BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """Returns (mean, ci_low, ci_high) via percentile bootstrap.

    Uses a seeded RNG so figures are byte-identical across runs. Skips NaN
    values; returns (nan, nan, nan) if fewer than two finite values are
    present (a CI from one sample is not meaningful).
    """
    finite = np.array([v for v in values if not np.isnan(v)], dtype=float)
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(finite.mean())
    if finite.size < 2:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, finite.size, size=(n, finite.size))
    resampled_means = finite[indices].mean(axis=1)
    lo = float(np.percentile(resampled_means, 100 * alpha / 2))
    hi = float(np.percentile(resampled_means, 100 * (1 - alpha / 2)))
    return mean, lo, hi


def _ppl_values(results: list[RunResult], strategy: str, budget: int) -> list[float]:
    return [r.perplexity for r in results if r.strategy == strategy and r.window_budget == budget]


def _kl_values(results: list[RunResult], strategy: str, budget: int) -> list[float]:
    return [
        r.kl_divergence for r in results if r.strategy == strategy and r.window_budget == budget
    ]


def _mean_ppl(results: list[RunResult], strategy: str, budget: int) -> float:
    vals = _ppl_values(results, strategy, budget)
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def _mean_kl(results: list[RunResult], strategy: str, budget: int) -> float:
    vals = _kl_values(results, strategy, budget)
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def _mean_ppl_delta(results: list[RunResult], strategy: str, budget: int) -> float:
    by_prompt = {
        r.prompt_idx: r.perplexity
        for r in results
        if r.strategy == "standard" and r.window_budget == budget
    }
    deltas = [
        r.perplexity - by_prompt[r.prompt_idx]
        for r in results
        if r.strategy == strategy and r.window_budget == budget and r.prompt_idx in by_prompt
    ]
    if not deltas:
        return float("nan")
    return float(np.mean(deltas))


def _mean_eviction_rate(
    results: list[RunResult],
    strategy: str,
    budget: int,
    layer_type: str,
) -> float:
    subset = [r for r in results if r.strategy == strategy and r.window_budget == budget]
    if not subset:
        return float("nan")
    if layer_type == "global":
        return float(np.mean([r.eviction_rate_global for r in subset]))
    return float(np.mean([r.eviction_rate_sliding for r in subset]))


def _total_sliding_bytes(results: list[RunResult], strategy: str, budget: int) -> float:
    subset = [r for r in results if r.strategy == strategy and r.window_budget == budget]
    if not subset:
        return float("nan")
    return float(np.mean([r.memory_bytes_sliding for r in subset]))


def plot_pareto(
    results: list[RunResult],
    output_path: Path,
    title: str = "Memory-Quality Pareto",
) -> None:
    """Figure 1: memory vs perplexity Pareto curves with bootstrap 95% CI bands."""
    strategies = sorted(set(r.strategy for r in results))
    budgets = sorted(set(r.window_budget for r in results))

    fig, ax = plt.subplots(figsize=(7, 5))
    for strategy in strategies:
        xs = [_total_sliding_bytes(results, strategy, b) / 1024 for b in budgets]
        ys: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        for b in budgets:
            mean, lo, hi = _bootstrap_ci(_ppl_values(results, strategy, b))
            ys.append(mean)
            lows.append(lo)
            highs.append(hi)
        color = STRATEGY_COLORS.get(strategy, "#111")
        ax.plot(xs, ys, marker="o", label=strategy, color=color, linewidth=1.8)
        ax.fill_between(xs, lows, highs, color=color, alpha=0.15, linewidth=0)

    ax.set_xlabel("Sliding-layer cache (KB)")
    ax.set_ylabel("Mean perplexity (95% CI)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_survival_heatmap(
    results: list[RunResult],
    output_path: Path,
    num_layers: int,
) -> None:
    """Figure 2: per-layer token survival rate heatmap, one subplot per strategy."""
    strategies = sorted(set(r.strategy for r in results))
    n = len(strategies)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, max(3, num_layers // 4)), sharey=True)
    if n == 1:
        axes = [axes]

    im = None
    for ax, strategy in zip(axes, strategies, strict=False):
        subset = [r for r in results if r.strategy == strategy]
        layer_rates: dict[int, list[float]] = {}
        for r in subset:
            for layer_idx, rate in r.token_survival_rates.items():
                layer_rates.setdefault(layer_idx, []).append(rate)

        matrix = np.array(
            [[np.mean(layer_rates.get(i, [1.0]))] for i in range(num_layers)]
        )  # [num_layers, 1]

        im = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="YlOrRd_r")
        ax.set_title(strategy, fontsize=10)
        ax.set_xlabel("mean")
        ax.set_xticks([0])
        ax.set_xticklabels(["survival"])
        ax.set_yticks(range(num_layers))
        ax.set_yticklabels([str(i) for i in range(num_layers)], fontsize=7)

    if im is not None:
        fig.colorbar(im, ax=axes[-1], label="survival rate")
    fig.suptitle("Per-layer token survival rate")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _total_cache_bytes(
    results: list[RunResult], strategy: str, budget: int, iso_memory: bool
) -> float:
    subset = _latency_subset(results, strategy, budget, iso_memory)
    if not subset:
        return float("nan")
    return float(np.mean([r.memory_bytes_sliding + r.memory_bytes_global for r in subset]))


def _latency_budget_values(results: list[RunResult]) -> tuple[list[int], bool]:
    iso_memory = any(r.memory_budget_mb > 0 for r in results)
    if iso_memory:
        return sorted(set(r.memory_budget_mb for r in results)), True
    return sorted(set(r.window_budget for r in results)), False


def _latency_subset(
    results: list[RunResult], strategy: str, budget: int, iso_memory: bool
) -> list[RunResult]:
    if iso_memory:
        return [r for r in results if r.strategy == strategy and r.memory_budget_mb == budget]
    return [r for r in results if r.strategy == strategy and r.window_budget == budget]


def _mean_decode_ms(
    results: list[RunResult], strategy: str, budget: int, iso_memory: bool
) -> float:
    subset = _latency_subset(results, strategy, budget, iso_memory)
    vals = [r.decode_ms_per_token for r in subset if r.decode_ms_per_token > 0]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def plot_latency_pareto(
    results: list[RunResult],
    output_path: Path,
    title: str = "Memory-Latency Pareto",
) -> None:
    """Figure: per-token decode latency vs total cache memory, one line per strategy."""
    strategies = sorted(set(r.strategy for r in results))
    budgets, iso_memory = _latency_budget_values(results)

    fig, ax = plt.subplots(figsize=(7, 5))
    plotted_any = False
    for strategy in strategies:
        xs = [_total_cache_bytes(results, strategy, b, iso_memory) / 1024 for b in budgets]
        ys = [_mean_decode_ms(results, strategy, b, iso_memory) for b in budgets]
        if all(np.isnan(y) for y in ys):
            continue
        color = STRATEGY_COLORS.get(strategy, "#111")
        ax.plot(xs, ys, marker="o", label=strategy, color=color, linewidth=1.8)
        plotted_any = True

    if not plotted_any:
        ax.text(
            0.5,
            0.5,
            "No latency data in these results",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    ax.set_xlabel("Total cache memory (KB)")
    ax.set_ylabel("Mean decode latency (ms / token)")
    ax.set_title(title)
    if plotted_any:
        ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_kl_pareto(
    results: list[RunResult],
    output_path: Path,
    title: str = "Memory-KL Divergence Pareto",
) -> None:
    """Figure: KL divergence vs sliding cache memory with bootstrap 95% CI bands."""
    strategies = [s for s in sorted(set(r.strategy for r in results)) if s != "standard"]
    budgets = sorted(set(r.window_budget for r in results))

    fig, ax = plt.subplots(figsize=(7, 5))
    for strategy in strategies:
        xs = [_total_sliding_bytes(results, strategy, b) / 1024 for b in budgets]
        ys: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        for b in budgets:
            mean, lo, hi = _bootstrap_ci(_kl_values(results, strategy, b))
            ys.append(mean)
            lows.append(lo)
            highs.append(hi)
        color = STRATEGY_COLORS.get(strategy, "#111")
        ax.plot(xs, ys, marker="o", label=strategy, color=color, linewidth=1.8)
        ax.fill_between(xs, lows, highs, color=color, alpha=0.15, linewidth=0)

    ax.set_xlabel("Sliding-layer cache (KB)")
    ax.set_ylabel("Mean KL divergence (95% CI)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_perplexity_delta(
    results: list[RunResult],
    output_path: Path,
    title: str = "Perplexity Delta vs Standard",
) -> None:
    """Diagnostic: mean perplexity delta from the standard baseline."""
    strategies = [s for s in sorted(set(r.strategy for r in results)) if s != "standard"]
    budgets = sorted(set(r.window_budget for r in results))

    fig, ax = plt.subplots(figsize=(7, 5))
    for strategy in strategies:
        ys = [_mean_ppl_delta(results, strategy, b) for b in budgets]
        color = STRATEGY_COLORS.get(strategy, "#111")
        ax.plot(budgets, ys, marker="o", label=strategy, color=color, linewidth=1.8)

    ax.axhline(0, color="#6b7280", linewidth=1, linestyle="--")
    ax.set_xlabel("Window budget")
    ax.set_ylabel("Mean perplexity - standard")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_kl_delta(
    results: list[RunResult],
    output_path: Path,
    title: str = "KL Divergence by Stress Budget",
) -> None:
    """Diagnostic: KL divergence from the standard baseline across budgets."""
    strategies = [s for s in sorted(set(r.strategy for r in results)) if s != "standard"]
    budgets = sorted(set(r.window_budget for r in results))

    fig, ax = plt.subplots(figsize=(7, 5))
    for strategy in strategies:
        ys = [_mean_kl(results, strategy, b) for b in budgets]
        color = STRATEGY_COLORS.get(strategy, "#111")
        ax.plot(budgets, ys, marker="o", label=strategy, color=color, linewidth=1.8)

    ax.set_xlabel("Window budget")
    ax.set_ylabel("Mean KL divergence from standard")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_eviction_pressure(
    results: list[RunResult],
    output_path: Path,
    title: str = "Eviction Pressure by Layer Type",
) -> None:
    """Diagnostic: sliding/global eviction rates by strategy and budget."""
    strategies = sorted(set(r.strategy for r in results))
    budgets = sorted(set(r.window_budget for r in results))

    fig, ax = plt.subplots(figsize=(8, 5))
    for strategy in strategies:
        color = STRATEGY_COLORS.get(strategy, "#111")
        sliding = [_mean_eviction_rate(results, strategy, b, "sliding") for b in budgets]
        global_ = [_mean_eviction_rate(results, strategy, b, "global") for b in budgets]
        ax.plot(budgets, sliding, marker="o", label=f"{strategy} sliding", color=color)
        if any(rate > 0 for rate in global_ if not np.isnan(rate)):
            ax.plot(
                budgets,
                global_,
                marker="x",
                linestyle="--",
                label=f"{strategy} global",
                color=color,
            )

    ax.set_xlabel("Window budget")
    ax.set_ylabel("Mean eviction rate")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_prompt_length_effect(
    results: list[RunResult],
    output_path: Path,
    title: str = "Prompt Length Effect on KL",
) -> None:
    """Diagnostic: prompt length vs KL divergence for non-standard strategies."""
    subset = [
        r
        for r in results
        if r.strategy != "standard" and r.prompt_tokens > 0 and r.kl_divergence >= 0
    ]
    fig, ax = plt.subplots(figsize=(7, 5))
    for strategy in sorted(set(r.strategy for r in subset)):
        runs = [r for r in subset if r.strategy == strategy]
        color = STRATEGY_COLORS.get(strategy, "#111")
        ax.scatter(
            [r.prompt_tokens for r in runs],
            [r.kl_divergence for r in runs],
            label=strategy,
            color=color,
            alpha=0.75,
        )

    ax.set_xlabel("Prompt tokens")
    ax.set_ylabel("KL divergence from standard")
    ax.set_title(title)
    if subset:
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_category_breakdown(
    results: list[RunResult],
    output_path: Path,
    target_budget: int,
) -> None:
    """Figure: per-category mean perplexity at a fixed budget, one bar group per category."""
    categories = sorted(set(r.category for r in results if r.category != "unknown"))
    if not categories:
        return
    strategies = [s for s in sorted(set(r.strategy for r in results)) if s != "standard"]
    subset = [r for r in results if r.window_budget == target_budget and r.strategy != "standard"]

    x = np.arange(len(categories))
    n = len(strategies)
    width = 0.8 / n

    fig, ax = plt.subplots(figsize=(max(8, 2 * len(categories)), 5))
    for i, strategy in enumerate(strategies):
        ppls: list[float] = []
        err_low: list[float] = []
        err_high: list[float] = []
        for cat in categories:
            cat_runs = [r for r in subset if r.strategy == strategy and r.category == cat]
            mean, lo, hi = _bootstrap_ci([r.perplexity for r in cat_runs])
            ppls.append(mean)
            # matplotlib expects positive offsets from the bar height
            err_low.append(0.0 if np.isnan(mean) else max(0.0, mean - lo))
            err_high.append(0.0 if np.isnan(mean) else max(0.0, hi - mean))
        offset = (i - n / 2 + 0.5) * width
        color = STRATEGY_COLORS.get(strategy, "#111")
        ax.bar(
            x + offset,
            ppls,
            width,
            label=strategy,
            color=color,
            alpha=0.85,
            yerr=[err_low, err_high],
            capsize=2,
            error_kw={"alpha": 0.6},
        )

    ax.set_xlabel("Prompt category")
    ax.set_ylabel("Mean perplexity (95% CI)")
    ax.set_title(f"Per-category mean perplexity at budget={target_budget}")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=15)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_layer_type_breakdown(
    results: list[RunResult],
    output_path: Path,
    layer_types: tuple[str, ...],
    target_budget: int,
) -> None:
    """Figure 3: at fixed budget, memory split by global vs sliding layers."""
    strategies = sorted(set(r.strategy for r in results if r.window_budget == target_budget))
    subset_full = [r for r in results if r.window_budget == target_budget]

    x = np.arange(len(strategies))
    width = 0.35

    sliding_mb = [
        float(
            np.mean(
                [r.memory_bytes_sliding / (1024 * 1024) for r in subset_full if r.strategy == s]
            )
        )
        for s in strategies
    ]
    global_mb = [
        float(
            np.mean(
                [r.memory_bytes_global / (1024 * 1024) for r in subset_full if r.strategy == s]
            )
        )
        for s in strategies
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        x,
        sliding_mb,
        width,
        label="Sliding layers",
        color="#60a5fa",
    )
    ax.bar(
        x,
        global_mb,
        width,
        bottom=sliding_mb,
        label="Global layers",
        color="#f59e0b",
    )

    ax.set_xlabel("Strategy")
    ax.set_ylabel("Mean cache memory (MB)")
    ax.set_title(f"Layer-type memory breakdown at budget={target_budget}")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=15)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
