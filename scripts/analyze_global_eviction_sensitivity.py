#!/usr/bin/env python3
"""Quantify Gemma 4's sensitivity to global-layer KV eviction.

Reads `results/gemma4_full_sweep/results.json` and asks:
when `hybrid_*` allocates the entire memory budget to global layers and forces
some eviction there, how much eviction does Gemma 4 tolerate before generation
collapses?

This is the per-prompt detail behind issue #75: at mb=64, `hybrid_h2o` evicts a
median of 348 global tokens and fails on 25/50 prompts; `pyramidkv` evicts 0
and fails on 0/50. The script characterises the eviction-count → failure-rate
relationship across the three hybrid_* strategies.

Output: a plain-text table written to
`results/gemma4_full_sweep/global_eviction_sensitivity.txt`
plus a printout of the same table to stdout.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

RESULTS_PATH = Path("results/gemma4_full_sweep/results.json")
OUTPUT_PATH = Path("results/gemma4_full_sweep/global_eviction_sensitivity.txt")

KL_BROKEN_THRESHOLD = 0.5  # KL above this = generation diverged
HYBRID_STRATEGIES = ("hybrid_h2o", "hybrid_snapkv", "hybrid_streaming")
REFERENCE_STRATEGY = "pyramidkv"  # the strategy that survives — for contrast


def load_runs(path: Path) -> list[dict]:
    with path.open() as f:
        return json.load(f)


def per_prompt_view(
    runs: list[dict], strategy: str, budget_mb: int
) -> list[tuple[int, int, float]]:
    """Return (prompt_idx, evicted_global_tokens, kl_divergence) per prompt."""
    rows: list[tuple[int, int, float]] = []
    for r in runs:
        if r["strategy"] != strategy or r["memory_budget_mb"] != budget_mb:
            continue
        rows.append((r["prompt_idx"], r["evicted_tokens_global"], r["kl_divergence"]))
    rows.sort(key=lambda row: row[1])  # sort by eviction count ascending
    return rows


def bin_by_eviction(
    rows: list[tuple[int, int, float]], edges: list[int]
) -> list[tuple[str, int, int, float]]:
    """Bin rows by global-eviction count. Each bin: (label, n, n_broken, median_kl)."""
    bins: dict[str, list[tuple[int, int, float]]] = defaultdict(list)
    for row in rows:
        evicted = row[1]
        label = "[0]"
        for i, edge in enumerate(edges):
            if evicted == 0:
                label = "[0]"
                break
            if evicted <= edge:
                lower = edges[i - 1] + 1 if i > 0 else 1
                label = f"[{lower}-{edge}]"
                break
        else:
            label = f"[>{edges[-1]}]"
        bins[label].append(row)
    out: list[tuple[str, int, int, float]] = []
    for label in (
        ["[0]"]
        + [f"[{edges[i - 1] + 1 if i > 0 else 1}-{edges[i]}]" for i in range(len(edges))]
        + [f"[>{edges[-1]}]"]
    ):
        if label not in bins:
            continue
        bin_rows = bins[label]
        n = len(bin_rows)
        n_broken = sum(1 for _, _, kl in bin_rows if kl > KL_BROKEN_THRESHOLD)
        med_kl = statistics.median([kl for _, _, kl in bin_rows])
        out.append((label, n, n_broken, med_kl))
    return out


def emit_strategy_section(
    strategy: str, budget_mb: int, rows: list[tuple[int, int, float]]
) -> str:
    out_lines: list[str] = []
    out_lines.append(f"\n## {strategy} at mb={budget_mb}\n")

    n_total = len(rows)
    n_broken = sum(1 for _, _, kl in rows if kl > KL_BROKEN_THRESHOLD)
    evicted_counts = [evicted for _, evicted, _ in rows]
    kls = [kl for _, _, kl in rows]
    out_lines.append(
        f"  prompts={n_total}  failures(KL>{KL_BROKEN_THRESHOLD})={n_broken}  "
        f"median_global_evict={statistics.median(evicted_counts)}  "
        f"max_global_evict={max(evicted_counts)}  "
        f"median_KL={statistics.median(kls):.4f}"
    )

    out_lines.append("\n  Eviction-bin → failure rate:")
    header = f"  {'bin':<14} {'n':>4} {'n_broken':>9} {'frac_broken':>12} {'median_kl':>10}"
    out_lines.append(header)
    for label, n, nb, mkl in bin_by_eviction(rows, [50, 200, 500, 1000]):
        frac = nb / n if n else 0.0
        out_lines.append(f"  {label:<14} {n:>4} {nb:>9} {frac:>11.1%} {mkl:>10.4f}")

    if n_total - n_broken > 0:
        max_safe = max(evicted for _, evicted, kl in rows if kl <= KL_BROKEN_THRESHOLD)
        out_lines.append(f"\n  largest non-failing run had {max_safe} global tokens evicted")
    if n_broken > 0:
        min_broken = min(evicted for _, evicted, kl in rows if kl > KL_BROKEN_THRESHOLD)
        out_lines.append(f"  smallest failing run had     {min_broken} global tokens evicted")
    return "\n".join(out_lines) + "\n"


def emit_reference_section(
    rows: list[tuple[int, int, float]], strategy: str, budget_mb: int
) -> str:
    n_total = len(rows)
    n_broken = sum(1 for _, _, kl in rows if kl > KL_BROKEN_THRESHOLD)
    evicted_counts = [evicted for _, evicted, _ in rows]
    return (
        f"\n## {strategy} at mb={budget_mb} (reference — recency rule + pyramid budget)\n"
        f"  prompts={n_total}  failures(KL>{KL_BROKEN_THRESHOLD})={n_broken}  "
        f"median_global_evict={statistics.median(evicted_counts)}  "
        f"max_global_evict={max(evicted_counts)}\n"
        f"  Conclusion: pyramid + recency does not pressure globals at this budget;\n"
        f"  with no global eviction Gemma 4 stays at standard quality.\n"
    )


def main() -> None:
    if not RESULTS_PATH.exists():
        raise SystemExit(f"missing input: {RESULTS_PATH}")
    runs = load_runs(RESULTS_PATH)

    sections: list[str] = []
    sections.append(
        "Global-eviction sensitivity on Gemma 4 (hybrid family, mb=64)\n"
        "=============================================================\n"
        f"source: {RESULTS_PATH}\n"
        f"failure threshold: KL > {KL_BROKEN_THRESHOLD}\n"
        "\n"
        "Question: at the budget cell where hybrid_h2o loses to pyramidkv, "
        "how does Gemma 4's failure rate scale with the number of global tokens "
        "the hybrid allocator forced to be evicted?\n"
    )

    for strategy in HYBRID_STRATEGIES:
        rows = per_prompt_view(runs, strategy, budget_mb=64)
        if not rows:
            sections.append(f"\n## {strategy} at mb=64\n  no runs found\n")
            continue
        sections.append(emit_strategy_section(strategy, 64, rows))

    ref_rows = per_prompt_view(runs, REFERENCE_STRATEGY, budget_mb=64)
    if ref_rows:
        sections.append(emit_reference_section(ref_rows, REFERENCE_STRATEGY, 64))

    body = "".join(sections)
    OUTPUT_PATH.write_text(body)
    print(body)
    print(f"\n[wrote {OUTPUT_PATH}]")


if __name__ == "__main__":
    main()
