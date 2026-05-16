#!/usr/bin/env python3
"""Validate a study experiment artifact.

Checks that the run actually exercised the new pipeline:

- new strategy taxonomy (no ``layertype_h2o``; canonical baselines present)
- iso-memory mode (``memory_budget_mb`` populated, ``window_budget`` zero)
- ``standard`` reference has KL == 0 and positive perplexity
- canonical baselines (``all_layers_*``) show meaningful perplexity degradation
  at the smallest budget — this is the "naive ports break on hybrid models"
  signal the study relies on
- per-token decode latency captured (``decode_ms_per_token`` > 0)
- all expected study figures rendered

Run:
    python scripts/validate_results.py --profile gemma4 \
                                      --results results/gemma4_full_sweep/results.json \
                                      --figures results/gemma4_full_sweep/figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CORE_EXPECTED_STRATEGIES = {
    "standard",
    "h2o",
    "snapkv",
    "streaming",
    "pyramidkv",
    "all_layers_h2o",
    "all_layers_snapkv",
    "all_layers_streaming",
    "all_layers_pyramidkv",
}

GEMMA4_EXPECTED_STRATEGIES = CORE_EXPECTED_STRATEGIES | {
    "proportional_h2o",
    "proportional_snapkv",
    "proportional_streaming",
    "proportional_pyramidkv",
}

EXPECTED_STRATEGIES_BY_PROFILE = {
    "core": CORE_EXPECTED_STRATEGIES,
    "gemma2": CORE_EXPECTED_STRATEGIES,
    "gemma4": GEMMA4_EXPECTED_STRATEGIES,
}

EXPECTED_STRATEGIES = CORE_EXPECTED_STRATEGIES

CANONICAL_BASELINES = {
    "all_layers_h2o",
    "all_layers_snapkv",
    "all_layers_streaming",
    "all_layers_pyramidkv",
}

EXPECTED_FIGURES = (
    "pareto.png",
    "kl_pareto.png",
    "latency_pareto.png",
    "survival_heatmap.png",
    "layer_type_breakdown.png",
    "category_breakdown.png",
)


def _load_runs(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def _run_strategies(runs: list[dict]) -> set[str]:
    return {r["strategy"] for r in runs}


def _smallest_budget(runs: list[dict]) -> int:
    return min(r["memory_budget_mb"] for r in runs if r["memory_budget_mb"] > 0)


def _mean_perplexity(runs: list[dict], strategy: str, budget: int) -> float:
    vals = [
        r["perplexity"]
        for r in runs
        if r["strategy"] == strategy and r["memory_budget_mb"] == budget
    ]
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def validate(results_path: Path, figures_dir: Path, profile: str = "core") -> list[str]:
    errors: list[str] = []

    runs = _load_runs(results_path)
    if not runs:
        return ["results file is empty"]
    if profile not in EXPECTED_STRATEGIES_BY_PROFILE:
        return [f"unknown validation profile: {profile}"]
    expected_strategies = EXPECTED_STRATEGIES_BY_PROFILE[profile]

    # Strategy taxonomy
    seen = _run_strategies(runs)
    missing = expected_strategies - seen
    if missing:
        errors.append(f"missing expected strategies: {sorted(missing)}")
    if "layertype_h2o" in seen:
        errors.append("found legacy 'layertype_h2o' — study runs should not use it")

    # Iso-memory mode
    if any(r["memory_budget_mb"] == 0 for r in runs):
        errors.append("some runs have memory_budget_mb=0 — study runs must use iso-memory")
    if any(r["window_budget"] != 0 for r in runs):
        errors.append("some runs have non-zero window_budget — study runs must use iso-memory")

    # Standard reference is well-formed
    standard_runs = [r for r in runs if r["strategy"] == "standard"]
    if not standard_runs:
        errors.append("no standard reference runs")
    else:
        if any(abs(r["kl_divergence"]) > 1e-9 for r in standard_runs):
            errors.append("standard runs have non-zero KL divergence")
        if any(r["perplexity"] <= 0 for r in standard_runs):
            errors.append("standard runs have non-positive perplexity")

    # Canonical baselines show degradation at smallest budget
    if standard_runs:
        positive_budgets = [r["memory_budget_mb"] for r in runs if r["memory_budget_mb"] > 0]
        if not positive_budgets:
            errors.append(
                "no runs have memory_budget_mb > 0 — cannot determine smallest iso-memory budget"
            )
        else:
            smallest = _smallest_budget(runs)
            std_ppl = _mean_perplexity(runs, "standard", smallest)
            if std_ppl != std_ppl:  # NaN
                errors.append(f"no standard runs at smallest budget {smallest}")
            else:
                for canonical in CANONICAL_BASELINES & seen:
                    ppl = _mean_perplexity(runs, canonical, smallest)
                    if ppl != ppl:
                        errors.append(f"{canonical}: no runs at smallest budget {smallest}")
                    elif ppl <= std_ppl * 1.5:
                        errors.append(
                            f"{canonical}: perplexity {ppl:.2f} at budget={smallest}MB is not "
                            f"meaningfully worse than standard {std_ppl:.2f} "
                            f"(expected canonical baselines to collapse on hybrid models)"
                        )

    # Latency captured
    if not any(r.get("prefill_ms", 0) > 0 for r in runs):
        errors.append("no runs have prefill_ms > 0 — latency tracking failed")
    if not any(r.get("decode_ms_per_token", 0) > 0 for r in runs):
        errors.append("no runs have decode_ms_per_token > 0 — latency tracking failed")

    # Figures rendered
    for figure in EXPECTED_FIGURES:
        path = figures_dir / figure
        if not path.exists():
            errors.append(f"missing figure: {figure}")
        elif path.stat().st_size == 0:
            errors.append(f"empty figure: {figure}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--figures", required=True, type=Path)
    parser.add_argument(
        "--profile",
        choices=sorted(EXPECTED_STRATEGIES_BY_PROFILE),
        default="core",
        help="strategy contract to enforce (core/gemma2=9 strategies, gemma4=13 strategies)",
    )
    args = parser.parse_args()

    errors = validate(args.results, args.figures, profile=args.profile)

    n_runs = len(_load_runs(args.results))
    n_strategies = len(_run_strategies(_load_runs(args.results)))
    print(f"runs={n_runs} strategies={n_strategies}")

    if errors:
        for e in errors:
            print(f"FAIL {e}", file=sys.stderr)
        return 1
    print("Validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
