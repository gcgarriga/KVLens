"""CLI entry points for KVLens."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
from rich.console import Console
from rich.table import Table

from kvlens.config import Gemma4Config, GenerationConfig
from kvlens.generation import generate
from kvlens.instrument import MetricsCollector
from kvlens.model import GemmaModel
from kvlens.tokenizer import GemmaTokenizer
from kvlens.weights import download_weight_snapshot, load_gemma2_weights, load_hf_weights

console = Console()


def estimate_model_memory_bytes(config: Gemma4Config) -> int:
    params = config.vocab_size * config.hidden_size
    params += config.vocab_size * config.num_layers * config.ple_dim
    params += config.hidden_size * config.num_layers * config.ple_dim

    for layer_idx in range(config.num_layers):
        layer_params = config.layer_params(layer_idx)
        q = config.hidden_size * config.num_heads * layer_params.head_dim
        k = config.hidden_size * config.num_kv_heads * layer_params.head_dim
        v = config.hidden_size * config.num_kv_heads * layer_params.head_dim
        o = config.num_heads * layer_params.head_dim * config.hidden_size
        mlp = (config.hidden_size * config.intermediate_size * 2) + (
            config.intermediate_size * config.hidden_size
        )
        ple = (config.hidden_size * config.ple_dim * 2) + config.hidden_size
        norms = (4 * config.hidden_size) + (2 * layer_params.head_dim) + 1
        params += q + k + v + o + mlp + ple + norms

    params += config.hidden_size
    return params * 2


def load_runtime(args: argparse.Namespace) -> tuple[GemmaModel, GemmaTokenizer]:
    repo_id = getattr(args, "repo_id", None)
    weights_path = getattr(args, "weights_path", None)
    tokenizer_model = getattr(args, "tokenizer_model", None)

    if repo_id is not None:
        console.print("[dim]Downloading / locating weights…[/dim]")
        try:
            weights_path = download_weight_snapshot(repo_id)
            tokenizer = GemmaTokenizer.from_pretrained(repo_id)
        except GatedRepoError as exc:
            raise RuntimeError(
                f"Cannot access gated repo '{repo_id}'. Authenticate with `hf auth login` "
                "using an account that has access, or pass local files with "
                "`--weights-path` and `--tokenizer-model`."
            ) from exc
        except RepositoryNotFoundError as exc:
            raise RuntimeError(f"Hugging Face repo '{repo_id}' was not found.") from exc
    elif weights_path is not None:
        weights_dir = Path(weights_path)
        if tokenizer_model is not None:
            tokenizer = GemmaTokenizer(tokenizer_model)
        elif weights_dir.is_dir():
            # Auto-detect tokenizer from the weights directory
            tokenizer = GemmaTokenizer.from_dir(weights_dir)
        else:
            raise RuntimeError(
                "Cannot find tokenizer. Provide --tokenizer-model or point "
                "--weights-path to a directory containing tokenizer.model or tokenizer.json"
            )
    else:
        raise RuntimeError(
            "Provide --repo-id for automatic download, or --weights-path to a local directory."
        )

    device_str = getattr(args, "device", None) or ("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[dim]Loading model on {device_str}…[/dim]")
    model_type = getattr(args, "model_type", "gemma4")
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with torch.device("meta"):
            if model_type == "gemma2":
                config = Gemma4Config.gemma2_2b()
                model = GemmaModel(config)
            else:
                model = GemmaModel(Gemma4Config())
    finally:
        torch.set_default_dtype(previous_dtype)
    if model_type == "gemma2":
        load_gemma2_weights(model, Path(weights_path), device=device_str)
    else:
        load_hf_weights(model, Path(weights_path), device=device_str)
    model.to(dtype=torch.bfloat16, device=device_str)
    model.lm_head.weight = model.embed_tokens.weight
    model.eval()
    return model, tokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kvlens",
        description="KV cache eviction research CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="command")

    generate_parser = subparsers.add_parser(
        "generate", help="Generate text with a configurable cache strategy"
    )
    generate_parser.add_argument("--prompt", required=True)
    generate_parser.add_argument("--repo-id")
    generate_parser.add_argument("--weights-path")
    generate_parser.add_argument("--tokenizer-model")
    generate_parser.add_argument("--model-type", default="gemma4", choices=["gemma4", "gemma2"])
    generate_parser.add_argument("--cache-strategy", default="standard")
    generate_parser.add_argument("--max-tokens", type=int, default=128)
    generate_parser.add_argument("--temperature", type=float, default=0.0)
    generate_parser.add_argument("--top-k", type=int, default=0)
    generate_parser.add_argument("--top-p", type=float, default=1.0)
    generate_parser.add_argument("--seed", type=int)
    generate_parser.add_argument("--instrument", action="store_true")
    generate_parser.add_argument("--device", help="torch device (default: cuda if available)")

    info_parser = subparsers.add_parser("info", help="Show model/config information")
    info_parser.add_argument("--tiny", action="store_true", help="Use the tiny CPU test config")

    experiment_parser = subparsers.add_parser("experiment", help="Run batch strategy comparison")
    experiment_parser.add_argument(
        "--prompts", required=True, help="Path to text file, one prompt per line"
    )
    experiment_parser.add_argument(
        "--strategies",
        default=(
            "standard,h2o,snapkv,streaming,pyramidkv,"
            "all_layers_h2o,all_layers_snapkv,all_layers_streaming,all_layers_pyramidkv,"
            "proportional_h2o,proportional_snapkv,proportional_streaming,proportional_pyramidkv"
        ),
    )
    budget_group = experiment_parser.add_mutually_exclusive_group()
    budget_group.add_argument(
        "--budgets",
        help="Iso-per-layer window sizes, comma-separated. Default: 64,128,256,512",
    )
    budget_group.add_argument(
        "--memory-budget",
        help=(
            "Iso-memory total cache budget in MB, comma-separated. Each strategy "
            "allocates this budget across layers per its policy. Mutually exclusive "
            "with --budgets."
        ),
    )
    experiment_parser.add_argument("--output", default="results.json")
    experiment_parser.add_argument("--repo-id")
    experiment_parser.add_argument("--weights-path")
    experiment_parser.add_argument("--tokenizer-model")
    experiment_parser.add_argument("--model-type", default="gemma4", choices=["gemma4", "gemma2"])
    experiment_parser.add_argument("--max-tokens", type=int, default=128)
    experiment_parser.add_argument(
        "--prompt-index-offset",
        type=int,
        default=0,
        help="Add this offset to prompt_idx in saved results, for chunked experiments.",
    )
    experiment_parser.add_argument("--temperature", type=float, default=0.0)
    experiment_parser.add_argument("--top-k", type=int, default=0)
    experiment_parser.add_argument("--top-p", type=float, default=1.0)
    experiment_parser.add_argument("--seed", type=int, default=42)
    experiment_parser.add_argument("--device", help="torch device (default: cuda if available)")

    plot_parser = subparsers.add_parser("plot", help="Generate study figures from results.json")
    plot_parser.add_argument("--input", required=True, help="Path to results.json")
    plot_parser.add_argument("--output-dir", default="figures", help="Directory for output PNGs")
    plot_parser.add_argument(
        "--model-type",
        default="gemma4",
        choices=["gemma4", "gemma2"],
        help="Model type (determines num_layers and layer_types)",
    )
    plot_parser.add_argument("--budget", type=int, default=256, help="Budget for Figure 3")

    return parser


def handle_generate(args: argparse.Namespace) -> int:
    model, tokenizer = load_runtime(args)
    metrics = MetricsCollector(enabled=args.instrument)
    generation_config = GenerationConfig(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        cache_strategy=args.cache_strategy,
    )
    input_ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long)

    result = generate(model, input_ids, generation_config, metrics=metrics)

    console.print(tokenizer.decode(result.sequences[0].tolist()))
    return 0


def _parse_prompts_file(path: Path) -> tuple[list[str], tuple[str, ...]]:
    """Parse prompts file. Lines may be prefixed 'category: prompt text'."""
    prompts, categories = [], []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if ": " in line:
            cat, prompt = line.split(": ", 1)
            categories.append(cat.strip())
            prompts.append(prompt.strip())
        else:
            categories.append("unknown")
            prompts.append(line)
    if not prompts:
        raise ValueError(f"No valid prompts found in {path}")
    return prompts, tuple(categories)


def _print_summary_table(
    results: list,
    strategies: list[str],
    budgets: list[int],
    iso_memory: bool,
) -> None:
    import numpy as np

    title = (
        "Mean perplexity by strategy × memory budget (MB)"
        if iso_memory
        else ("Mean perplexity by strategy × budget")
    )
    label = "mem_mb" if iso_memory else "budget"
    table = Table(title=title)
    table.add_column("strategy", style="bold")
    for b in budgets:
        table.add_column(f"{label}={b}", justify="right")
    for s in strategies:
        row = [s]
        for b in budgets:
            if iso_memory:
                vals = [
                    r.perplexity for r in results if r.strategy == s and r.memory_budget_mb == b
                ]
            else:
                vals = [r.perplexity for r in results if r.strategy == s and r.window_budget == b]
            row.append(f"{np.mean(vals):.2f}" if vals else "—")
        table.add_row(*row)
    console.print(table)


def handle_experiment(args: argparse.Namespace) -> int:
    from kvlens.experiments.runner import ExperimentConfig, run_experiment, save_results

    prompts, categories = _parse_prompts_file(Path(args.prompts))
    strategies = [s.strip() for s in args.strategies.split(",")]

    iso_memory = args.memory_budget is not None
    if iso_memory:
        budgets = [int(b.strip()) for b in args.memory_budget.split(",")]
    else:
        budgets_str = args.budgets if args.budgets is not None else "64,128,256,512"
        budgets = [int(b.strip()) for b in budgets_str.split(",")]

    model, tokenizer = load_runtime(args)
    exp_config = ExperimentConfig(
        prompts=prompts,
        strategies=strategies,
        window_budgets=() if iso_memory else tuple(budgets),
        memory_budgets_mb=tuple(budgets) if iso_memory else (),
        max_tokens=args.max_tokens,
        seed=42 if args.seed is None else args.seed,
        categories=categories,
        prompt_index_offset=args.prompt_index_offset,
    )
    mode_label = "MB iso-memory" if iso_memory else "iso-per-layer"
    console.print(
        f"[dim]Running {len(prompts)} prompts × {len(strategies)} strategies"
        f" × {len(budgets)} {mode_label} budgets…[/dim]"
    )
    results = run_experiment(model, tokenizer, exp_config)
    output_path = Path(args.output)
    save_results(results, output_path)
    console.print(f"Results saved to [bold]{output_path}[/bold] ({len(results)} runs)")
    _print_summary_table(results, strategies, budgets, iso_memory)
    return 0


def handle_plot(args: argparse.Namespace) -> int:
    try:
        from kvlens.experiments.plot import (
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
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "The 'plot' command requires optional plotting dependencies. "
            'Install them with `pip install "kvlens[dev]"` or install `matplotlib`.'
        ) from exc
    from kvlens.experiments.runner import load_results

    results = load_results(Path(args.input))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = Gemma4Config.gemma2_2b() if args.model_type == "gemma2" else Gemma4Config()

    plot_pareto(results, output_dir / "pareto.png")
    console.print(f"[dim]Wrote {output_dir}/pareto.png[/dim]")

    plot_kl_pareto(results, output_dir / "kl_pareto.png")
    console.print(f"[dim]Wrote {output_dir}/kl_pareto.png[/dim]")

    plot_latency_pareto(results, output_dir / "latency_pareto.png")
    console.print(f"[dim]Wrote {output_dir}/latency_pareto.png[/dim]")

    plot_survival_heatmap(
        results, output_dir / "survival_heatmap.png", num_layers=config.num_layers
    )
    console.print(f"[dim]Wrote {output_dir}/survival_heatmap.png[/dim]")

    plot_layer_type_breakdown(
        results,
        output_dir / "layer_type_breakdown.png",
        layer_types=config.layer_types,
        target_budget=args.budget,
    )
    console.print(f"[dim]Wrote {output_dir}/layer_type_breakdown.png[/dim]")

    plot_category_breakdown(
        results, output_dir / "category_breakdown.png", target_budget=args.budget
    )
    console.print(f"[dim]Wrote {output_dir}/category_breakdown.png[/dim]")

    plot_perplexity_delta(results, output_dir / "perplexity_delta.png")
    console.print(f"[dim]Wrote {output_dir}/perplexity_delta.png[/dim]")

    plot_kl_delta(results, output_dir / "kl_delta.png")
    console.print(f"[dim]Wrote {output_dir}/kl_delta.png[/dim]")

    plot_eviction_pressure(results, output_dir / "eviction_pressure.png")
    console.print(f"[dim]Wrote {output_dir}/eviction_pressure.png[/dim]")

    plot_prompt_length_effect(results, output_dir / "prompt_length_effect.png")
    console.print(f"[dim]Wrote {output_dir}/prompt_length_effect.png[/dim]")
    return 0


def handle_info(args: argparse.Namespace) -> int:
    config = Gemma4Config.tiny() if args.tiny else Gemma4Config()
    memory_bytes = estimate_model_memory_bytes(config)
    table = Table(title="Model Info")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("num_layers", str(config.num_layers))
    table.add_row("hidden_size", str(config.hidden_size))
    table.add_row("num_heads", str(config.num_heads))
    table.add_row("num_kv_heads", str(config.num_kv_heads))
    table.add_row("estimated_bfloat16_bytes", str(memory_bytes))
    console.print(table)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command is None:
            parser.print_help()
            return 0
        if args.command == "generate":
            return handle_generate(args)
        if args.command == "info":
            return handle_info(args)
        if args.command == "experiment":
            return handle_experiment(args)
        if args.command == "plot":
            return handle_plot(args)
        raise ValueError(f"unknown command '{args.command}'")
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        return 1
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
        return 130
