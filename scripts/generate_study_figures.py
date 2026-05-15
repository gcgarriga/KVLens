#!/usr/bin/env python3
"""Produce study figures for the Gemma 4 KV-cache eviction case study.

Figures:

  F1: KL Pareto, Gemma 4 sweep
  F2: Cache state at mb=512 — memory-equivalent cache, bimodal KL
  F3: Length-threshold ablation
  F4: Eviction-count vs KL at mb=64 — empty band between zero and high-KL evictions
  F5: Gemma 2 comparison

Each figure is written as both a PDF (vector) and a PNG (300 DPI). Output:
``study/figures/``.

The script is deterministic and zero-compute — it reads the merged result
JSONs and produces the figures locally. Re-run after any change to the
underlying results to regenerate.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path("study/figures")

GEMMA4_SWEEP = Path("results/gemma4_full_sweep/results.json")
GEMMA4_LENGTH_ABLATION_DIR = Path("results/gemma4_length_ablation")
GEMMA2_SCOPE_CHECK = Path("results/gemma2_comparison/results.json")

WORKING = ("standard", "pyramidkv", "hybrid_h2o", "hybrid_snapkv", "hybrid_streaming")
DIVERGENT_HYBRID_ADAPTED = ("h2o", "snapkv", "streaming")
DIVERGENT_ALL_LAYERS = (
    "all_layers_h2o",
    "all_layers_snapkv",
    "all_layers_streaming",
    "all_layers_pyramidkv",
)
DIVERGENT_PROPORTIONAL = (
    "proportional_h2o",
    "proportional_snapkv",
    "proportional_streaming",
    "proportional_pyramidkv",
)
DIVERGENT_ALL = DIVERGENT_HYBRID_ADAPTED + DIVERGENT_ALL_LAYERS + DIVERGENT_PROPORTIONAL

COLOR_WORKING = "#1f77b4"
COLOR_HYBRID = "#2ca02c"
COLOR_DIVERGENT = "#d62728"
COLOR_REFERENCE = "#000000"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def load(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"missing input: {path}")
    with path.open() as f:
        return json.load(f)


def median_by(runs: list[dict], strategy: str, budget_mb: int, key: str) -> float:
    vals = [
        r[key] for r in runs if r["strategy"] == strategy and r["memory_budget_mb"] == budget_mb
    ]
    return statistics.median(vals) if vals else float("nan")


def save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{name}.pdf")
    fig.savefig(OUT_DIR / f"{name}.png", dpi=300)
    print(f"  wrote {OUT_DIR / name}.pdf and .png")


def normalize_readme_highlights() -> None:
    """Create README highlight PNGs with matched title and plot geometry."""
    required = ("f1_kl_pareto.png", "f2_memory_equivalent_cache_mb512.png")
    if not all((OUT_DIR / name).exists() for name in required):
        print(
            "README highlight normalization skipped "
            "(F1/F2 outputs missing; needs results/gemma4_full_sweep/results.json, "
            "regenerate with scripts/run_gemma4_sweep.sh)"
        )
        return
    canvas_size = (2105, 1400)
    target_axes = (330, 235, 1843, 1035)
    title_color = "#141414"

    specs = (
        {
            "source": "f1_kl_pareto.png",
            "output": "readme_f1_kl_pareto.png",
            "title": "KL Pareto across memory budgets - Gemma 4",
            "crop": (0, 65, 1814, 1137),
            "axes": (151, 9, 1664, 935),
        },
        {
            "source": "f2_memory_equivalent_cache_mb512.png",
            "output": "readme_f2_memory_equivalent_cache_mb512.png",
            "title": "Same memory, different quality at mb=512",
            "crop": (0, 65, 2345, 1446),
            "axes": (300, 9, 2046, 1051),
        },
    )

    target_width = target_axes[2] - target_axes[0]
    target_height = target_axes[3] - target_axes[1]
    for spec in specs:
        source = plt.imread(OUT_DIR / str(spec["source"]))
        crop_left, crop_top, crop_right, crop_bottom = spec["crop"]
        crop = source[crop_top:crop_bottom, crop_left:crop_right]
        axis_left, axis_top, axis_right, axis_bottom = spec["axes"]
        scale_x = target_width / (axis_right - axis_left)
        scale_y = target_height / (axis_bottom - axis_top)

        resized_width = round(crop.shape[1] * scale_x)
        resized_height = round(crop.shape[0] * scale_y)
        paste_left = target_axes[0] - axis_left * scale_x
        paste_top = target_axes[1] - axis_top * scale_y

        fig = plt.figure(figsize=(canvas_size[0] / 100, canvas_size[1] / 100), dpi=100)
        fig.patch.set_facecolor("white")
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set_xlim(0, canvas_size[0])
        ax.set_ylim(canvas_size[1], 0)
        ax.axis("off")
        ax.imshow(
            crop,
            extent=(
                paste_left,
                paste_left + resized_width,
                paste_top + resized_height,
                paste_top,
            ),
            interpolation="lanczos",
        )
        ax.text(
            canvas_size[0] / 2,
            92,
            str(spec["title"]),
            ha="center",
            va="center",
            color=title_color,
            family="DejaVu Serif",
            fontsize=23,
        )
        fig.savefig(OUT_DIR / str(spec["output"]), dpi=100, bbox_inches=None, pad_inches=0)
        plt.close(fig)


def figure_1_kl_pareto(runs: list[dict] | None) -> None:
    if runs is None:
        print(f"F1 skipped (missing {GEMMA4_SWEEP}; regenerate with scripts/run_gemma4_sweep.sh)")
        return
    print("F1 KL Pareto across budgets")
    budgets = sorted({r["memory_budget_mb"] for r in runs})
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for strategy in WORKING:
        kls = [median_by(runs, strategy, b, "kl_divergence") for b in budgets]
        color = (
            COLOR_REFERENCE
            if strategy == "standard"
            else (COLOR_WORKING if strategy == "pyramidkv" else COLOR_HYBRID)
        )
        marker = "o" if strategy == "standard" else ("s" if strategy == "pyramidkv" else "^")
        ax.plot(budgets, kls, marker=marker, color=color, label=strategy, linewidth=1.5)
    for strategy in DIVERGENT_ALL:
        kls = [median_by(runs, strategy, b, "kl_divergence") for b in budgets]
        ax.plot(budgets, kls, marker=".", color=COLOR_DIVERGENT, alpha=0.4, linewidth=0.8)
    ax.plot(
        [],
        [],
        marker=".",
        color=COLOR_DIVERGENT,
        alpha=0.6,
        label="divergent families (n=11)",
    )

    ax.set_xscale("log", base=2)
    ax.set_xticks(budgets)
    ax.set_xticklabels([str(b) for b in budgets])
    ax.set_xlabel("memory budget (MB, log scale)")
    ax.set_ylabel("median KL divergence vs standard")
    ax.set_title("KL Pareto across memory budgets — Gemma 4 (50 prompts × 16 strategies)")
    ax.grid(True, which="both", alpha=0.25, linestyle=":")
    ax.legend(loc="center right", framealpha=0.95)
    save(fig, "f1_kl_pareto")
    plt.close(fig)


def figure_2_memory_equivalent_cache(runs: list[dict] | None) -> None:
    if runs is None:
        print(f"F2 skipped (missing {GEMMA4_SWEEP}; regenerate with scripts/run_gemma4_sweep.sh)")
        return
    print("F2 memory-equivalent cache at mb=512 with bimodal KL")
    mb = 512
    strategies = sorted({r["strategy"] for r in runs})
    rows: list[tuple[str, float, float, str]] = []
    for s in strategies:
        kl = median_by(runs, s, mb, "kl_divergence")
        mem_total = (
            median_by(runs, s, mb, "memory_bytes_sliding")
            + median_by(runs, s, mb, "memory_bytes_global")
        ) / 1e6
        family = "working" if s in WORKING else ("divergent" if s in DIVERGENT_ALL else "other")
        rows.append((s, kl, mem_total, family))
    rows.sort(key=lambda r: (r[3] == "divergent", r[1]))

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    xs = np.arange(len(rows))
    bar_colors = [
        COLOR_REFERENCE
        if r[0] == "standard"
        else (
            COLOR_HYBRID
            if r[0].startswith("hybrid_")
            else (COLOR_WORKING if r[0] == "pyramidkv" else COLOR_DIVERGENT)
        )
        for r in rows
    ]
    bars = ax.bar(xs, [r[1] for r in rows], color=bar_colors, edgecolor="black", linewidth=0.4)
    for bar, (_s, kl, mem, _) in zip(bars, rows, strict=True):
        if kl > 1.0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                kl + 0.3,
                f"{mem:.0f}MB",
                ha="center",
                va="bottom",
                fontsize=7,
                color="dimgray",
            )
    ax.set_xticks(xs)
    ax.set_xticklabels([r[0] for r in rows], rotation=60, ha="right")
    ax.set_ylabel("median KL divergence vs standard")
    ax.set_title(
        "Memory-equivalent cache at mb=512: every strategy fits the same memory, "
        "but quality is bimodal"
    )
    ax.axhline(
        0.05,
        color="gray",
        linestyle=":",
        linewidth=0.8,
        label="KL = 0.05 (acceptance gate)",
    )
    ax.grid(True, axis="y", alpha=0.25, linestyle=":")
    ax.legend(loc="upper left", framealpha=0.9)

    note = (
        "Annotations show median actual cache memory (MB). All strategies "
        "converge to ~50 MB at mb=512; bimodality is in selection, not size."
    )
    ax.text(
        0.5,
        -0.4,
        note,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
        style="italic",
        color="dimgray",
    )
    save(fig, "f2_memory_equivalent_cache_mb512")
    plt.close(fig)


def figure_3_length_threshold() -> None:
    print("F3 length-threshold ablation")
    panel_data: dict[str, list[dict]] = {}
    panels = (
        ("256", "length_256.json"),
        ("1k", "length_1k.json"),
        ("2k", "length_2k.json"),
    )
    for label, fname in panels:
        path = GEMMA4_LENGTH_ABLATION_DIR / fname
        panel_data[label] = load(path)

    strategies = (
        "standard",
        "pyramidkv",
        "hybrid_h2o",
        "hybrid_snapkv",
        "hybrid_streaming",
        "h2o",
        "snapkv",
        "streaming",
    )
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.0), sharey=True)
    for ax, (label, runs) in zip(axes, panel_data.items(), strict=True):
        kls = []
        colors = []
        for s in strategies:
            ks = [r["kl_divergence"] for r in runs if r["strategy"] == s]
            kls.append(statistics.median(ks) if ks else 0.0)
            colors.append(
                COLOR_REFERENCE
                if s == "standard"
                else (
                    COLOR_HYBRID
                    if s.startswith("hybrid_")
                    else (COLOR_WORKING if s == "pyramidkv" else COLOR_DIVERGENT)
                )
            )
        ax.bar(range(len(strategies)), kls, color=colors, edgecolor="black", linewidth=0.4)
        ax.set_xticks(range(len(strategies)))
        ax.set_xticklabels(strategies, rotation=60, ha="right")
        ax.set_title(f"prompt length ≈ {label}")
        ax.set_ylim(0, 17)
        ax.axhline(0.05, color="gray", linestyle=":", linewidth=0.8)
        ax.grid(True, axis="y", alpha=0.25, linestyle=":")
    axes[0].set_ylabel("median KL divergence vs standard")
    fig.suptitle(
        "Sliding-window engagement is the trigger: at length ≤ 256 every "
        "strategy reduces to standard; ≥ 1k splits bimodally"
    )
    save(fig, "f3_length_threshold_ablation")
    plt.close(fig)


def figure_4_eviction_threshold(runs: list[dict] | None) -> None:
    if runs is None:
        print(f"F4 skipped (missing {GEMMA4_SWEEP}; regenerate with scripts/run_gemma4_sweep.sh)")
        return
    print("F4 eviction-count vs KL at mb=64 — empty band between zero and high-KL evictions")
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.0), sharex=True, sharey=True)
    for ax, strategy in zip(
        axes, ("hybrid_h2o", "hybrid_snapkv", "hybrid_streaming"), strict=True
    ):
        rows = [
            (r["evicted_tokens_global"], r["kl_divergence"])
            for r in runs
            if r["strategy"] == strategy and r["memory_budget_mb"] == 64
        ]
        xs = np.array([r[0] for r in rows])
        ys = np.array([r[1] for r in rows])
        divergent = ys > 0.5
        ax.scatter(
            xs[~divergent],
            ys[~divergent],
            color=COLOR_HYBRID,
            edgecolor="black",
            linewidth=0.4,
            s=44,
            label=f"reference-preserving ({(~divergent).sum()})",
            zorder=3,
        )
        ax.scatter(
            xs[divergent],
            ys[divergent],
            color=COLOR_DIVERGENT,
            edgecolor="black",
            linewidth=0.4,
            s=44,
            label=f"high-KL ({divergent.sum()})",
            zorder=3,
        )
        ax.axvspan(1, 695, alpha=0.12, color="gray", label="empty bin (no graceful regime)")
        ax.set_xlabel("global tokens evicted (per prompt)")
        ax.set_title(strategy)
        ax.grid(True, alpha=0.25, linestyle=":")
        ax.set_xlim(-50, max(xs.max(), 1500) * 1.05)
        ax.legend(loc="center right", framealpha=0.9, fontsize=8)
    axes[0].set_ylabel("KL divergence vs standard")
    fig.suptitle(
        "Gemma 4 global-eviction sensitivity at mb=64: "
        "empty band [1, 695] across 150 runs (3 hybrid × 50 prompts)"
    )
    save(fig, "f4_global_eviction_threshold")
    plt.close(fig)


def figure_5_gemma2_scope_check() -> None:
    if not GEMMA2_SCOPE_CHECK.exists():
        print("F5 skipped (Gemma 2 scope-check results not present)")
        return
    print("F5 Gemma 2 comparison")
    runs = load(GEMMA2_SCOPE_CHECK)
    strategies = ("standard", "h2o", "hybrid_h2o")
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    xs = np.arange(len(strategies))
    kls = [
        statistics.median([r["kl_divergence"] for r in runs if r["strategy"] == s])
        for s in strategies
    ]
    colors = [COLOR_REFERENCE, COLOR_DIVERGENT, COLOR_HYBRID]
    bars = ax.bar(xs, kls, color=colors, edgecolor="black", linewidth=0.4)
    for bar, kl in zip(bars, kls, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            kl + 0.0005,
            f"{kl:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(strategies)
    ax.set_ylabel("median KL divergence vs standard")
    ax.set_title("Gemma 2 scope check (mb=256, prompts ≈ 7.7k tokens)")
    ax.set_ylim(0, max(kls + [0.05]) * 1.4)
    ax.axhline(0.05, color="gray", linestyle=":", linewidth=0.8, label="KL = 0.05 gate")
    ax.grid(True, axis="y", alpha=0.25, linestyle=":")
    ax.legend(framealpha=0.9)
    fig.text(
        0.5,
        -0.06,
        "h2o engages 47k sliding evictions per prompt yet stays at standard quality.",
        ha="center",
        va="top",
        fontsize=8,
        style="italic",
        color="dimgray",
    )
    save(fig, "f5_gemma2_scope_check")
    plt.close(fig)


def main() -> None:
    print(f"output → {OUT_DIR}/")
    gemma4_sweep = load(GEMMA4_SWEEP) if GEMMA4_SWEEP.exists() else None
    figure_1_kl_pareto(gemma4_sweep)
    figure_2_memory_equivalent_cache(gemma4_sweep)
    normalize_readme_highlights()
    figure_3_length_threshold()
    figure_4_eviction_threshold(gemma4_sweep)
    figure_5_gemma2_scope_check()
    print("\ndone.")


if __name__ == "__main__":
    main()
