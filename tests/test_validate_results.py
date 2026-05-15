"""Tests for scripts/validate_results.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_validator():
    """Load the validator module without making it a package."""
    src = Path(__file__).resolve().parent.parent / "scripts" / "validate_results.py"
    spec = importlib.util.spec_from_file_location("validate_results", src)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VALIDATOR = _load_validator()


def _make_run(strategy: str, memory_budget_mb: int, **kwargs) -> dict:
    base = {
        "strategy": strategy,
        "window_budget": 0,
        "memory_budget_mb": memory_budget_mb,
        "prompt_idx": 0,
        "kl_divergence": 0.5 if strategy != "standard" else 0.0,
        "perplexity": 5.0,
        "memory_bytes_sliding": 1024,
        "memory_bytes_global": 2048,
        "prefill_ms": 100.0,
        "decode_ms_per_token": 5.0,
        "total_decode_ms": 50.0,
    }
    base.update(kwargs)
    return base


def _make_artifact(tmp_path: Path, runs: list[dict]) -> tuple[Path, Path]:
    results = tmp_path / "results.json"
    figures = tmp_path / "figures"
    figures.mkdir()
    for fname in VALIDATOR.EXPECTED_FIGURES:
        (figures / fname).write_bytes(b"\x89PNG\r\n\x1a\n")
    results.write_text(json.dumps(runs))
    return results, figures


def _make_passing_runs() -> list[dict]:
    """Construct a results dataset that should pass all validator checks."""
    runs: list[dict] = []
    smallest = 1
    other_budgets = [2, 4, 8, 16]
    for budget in (smallest, *other_budgets):
        runs.append(_make_run("standard", budget, kl_divergence=0.0, perplexity=2.0))
        for hybrid in ("h2o", "snapkv", "streaming", "pyramidkv"):
            runs.append(_make_run(hybrid, budget, perplexity=2.5))
        for proportional in (
            "proportional_h2o",
            "proportional_snapkv",
            "proportional_streaming",
            "proportional_pyramidkv",
        ):
            runs.append(_make_run(proportional, budget, perplexity=2.5))
        for canonical in (
            "all_layers_h2o",
            "all_layers_snapkv",
            "all_layers_streaming",
            "all_layers_pyramidkv",
        ):
            # Canonical baselines collapse at the smallest budget — exactly
            # what the study needs to demonstrate.
            ppl = 50.0 if budget == smallest else 5.0
            runs.append(_make_run(canonical, budget, perplexity=ppl))
    return runs


def test_passing_artifact_has_no_errors(tmp_path: Path) -> None:
    results, figures = _make_artifact(tmp_path, _make_passing_runs())
    assert VALIDATOR.validate(results, figures) == []


def test_missing_canonical_baseline_flagged(tmp_path: Path) -> None:
    runs = [r for r in _make_passing_runs() if r["strategy"] != "all_layers_snapkv"]
    results, figures = _make_artifact(tmp_path, runs)
    errors = VALIDATOR.validate(results, figures)
    assert any("all_layers_snapkv" in e for e in errors)


def test_gemma4_profile_requires_proportional_strategies(tmp_path: Path) -> None:
    runs = [r for r in _make_passing_runs() if r["strategy"] != "proportional_h2o"]
    results, figures = _make_artifact(tmp_path, runs)

    assert VALIDATOR.validate(results, figures) == []

    errors = VALIDATOR.validate(results, figures, profile="gemma4")
    assert any("proportional_h2o" in e for e in errors)


def test_legacy_layertype_h2o_flagged(tmp_path: Path) -> None:
    runs = _make_passing_runs() + [_make_run("layertype_h2o", 1)]
    results, figures = _make_artifact(tmp_path, runs)
    errors = VALIDATOR.validate(results, figures)
    assert any("layertype_h2o" in e for e in errors)


def test_iso_per_layer_mode_flagged(tmp_path: Path) -> None:
    runs = _make_passing_runs()
    runs[0]["window_budget"] = 32  # accidentally iso-per-layer
    results, figures = _make_artifact(tmp_path, runs)
    errors = VALIDATOR.validate(results, figures)
    assert any("window_budget" in e for e in errors)


def test_canonical_baseline_not_collapsing_flagged(tmp_path: Path) -> None:
    runs = _make_passing_runs()
    # Set every all_layers_h2o perplexity equal to standard — no degradation.
    for r in runs:
        if r["strategy"] == "all_layers_h2o":
            r["perplexity"] = 2.0
    results, figures = _make_artifact(tmp_path, runs)
    errors = VALIDATOR.validate(results, figures)
    assert any("all_layers_h2o" in e and "not meaningfully worse" in e for e in errors)


def test_zero_latency_flagged(tmp_path: Path) -> None:
    runs = _make_passing_runs()
    for r in runs:
        r["prefill_ms"] = 0.0
        r["decode_ms_per_token"] = 0.0
    results, figures = _make_artifact(tmp_path, runs)
    errors = VALIDATOR.validate(results, figures)
    assert any("prefill_ms" in e for e in errors)
    assert any("decode_ms_per_token" in e for e in errors)


def test_missing_figure_flagged(tmp_path: Path) -> None:
    results, figures = _make_artifact(tmp_path, _make_passing_runs())
    (figures / "latency_pareto.png").unlink()
    errors = VALIDATOR.validate(results, figures)
    assert any("latency_pareto.png" in e for e in errors)


def test_empty_results_flagged(tmp_path: Path) -> None:
    results, figures = _make_artifact(tmp_path, [])
    errors = VALIDATOR.validate(results, figures)
    assert errors == ["results file is empty"]


def test_missing_positive_memory_budget_flagged(tmp_path: Path) -> None:
    runs = _make_passing_runs()
    for run in runs:
        run["memory_budget_mb"] = 0
    results, figures = _make_artifact(tmp_path, runs)
    errors = VALIDATOR.validate(results, figures)
    assert any("memory_budget_mb > 0" in e for e in errors)


@pytest.mark.parametrize("strategy", sorted(VALIDATOR.EXPECTED_STRATEGIES))
def test_each_expected_strategy_must_be_present(tmp_path: Path, strategy: str) -> None:
    runs = [r for r in _make_passing_runs() if r["strategy"] != strategy]
    results, figures = _make_artifact(tmp_path, runs)
    errors = VALIDATOR.validate(results, figures)
    assert any(strategy in e for e in errors)
