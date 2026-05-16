"""Batch experiment runner for KV cache strategy comparison."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from kvlens.cache import create_cache
from kvlens.config import GenerationConfig
from kvlens.experiments.metrics import perplexity
from kvlens.generation import decode_step, generate, prefill

if TYPE_CHECKING:
    from kvlens.model import GemmaModel
    from kvlens.tokenizer import GemmaTokenizer


@dataclass(frozen=True)
class ExperimentConfig:
    prompts: list[str]
    strategies: list[str]
    window_budgets: tuple[int, ...] = ()
    memory_budgets_mb: tuple[int, ...] = ()
    max_tokens: int = 128
    seed: int = 42
    categories: tuple[str, ...] = ()
    prompt_index_offset: int = 0

    def __post_init__(self) -> None:
        # Exactly one budget mode must be active so each (strategy, budget, prompt)
        # combination is unambiguous.
        if bool(self.window_budgets) == bool(self.memory_budgets_mb):
            raise ValueError(
                "Provide exactly one of window_budgets or memory_budgets_mb (mutually exclusive)"
            )
        if self.prompt_index_offset < 0:
            raise ValueError("prompt_index_offset must be non-negative")


@dataclass
class RunResult:
    strategy: str
    window_budget: int = 0  # 0 in iso-memory mode
    memory_budget_mb: int = 0  # 0 in iso-per-layer mode
    prompt_idx: int = 0
    kl_divergence: float = 0.0
    perplexity: float = 0.0
    memory_bytes_sliding: int = 0
    memory_bytes_global: int = 0
    token_survival_rates: dict[int, float] = field(default_factory=dict)
    category: str = "unknown"
    prompt_tokens: int = 0
    generated_tokens: int = 0
    effective_tokens: int = 0
    evicted_tokens_sliding: int = 0
    evicted_tokens_global: int = 0
    eviction_rate_sliding: float = 0.0
    eviction_rate_global: float = 0.0
    cache_lengths_by_layer: dict[int, int] = field(default_factory=dict)
    evictions_by_layer: dict[int, int] = field(default_factory=dict)
    prefill_ms: float = 0.0
    total_decode_ms: float = 0.0
    decode_ms_per_token: float = 0.0


@dataclass
class _ReplayResult:
    logits: torch.Tensor
    prefill_ms: float
    total_decode_ms: float
    decode_count: int


def _kl_divergence(reference_logits: torch.Tensor, candidate_logits: torch.Tensor) -> float:
    """KL(reference || candidate) averaged over all positions."""
    ref_probs = torch.softmax(reference_logits.float(), dim=-1)
    ref_log = torch.log_softmax(reference_logits.float(), dim=-1)
    cand_log = torch.log_softmax(candidate_logits.float(), dim=-1)
    return float((ref_probs * (ref_log - cand_log)).sum(dim=-1).mean().item())


def _sync_if_cuda(device: torch.device) -> None:
    """Force CUDA work to complete so wall-clock timings reflect real GPU work."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _replay_forced_tokens(
    model: GemmaModel,
    input_ids: torch.Tensor,
    forced_tokens: list[int],
    strategy: str,
    window_budget: int | None,
    memory_budget_bytes: int | None,
    dtype_bytes: int,
) -> _ReplayResult:
    """Run teacher-forced decoding with the given cache strategy.

    Returns logits [1, T, vocab] where T = len(forced_tokens), plus prefill and
    decode wall-clock timings (ms).
    """
    cache = create_cache(
        strategy,
        model.config,
        window_budget=window_budget,
        memory_budget_bytes=memory_budget_bytes,
        dtype_bytes=dtype_bytes,
    )
    logits: list[torch.Tensor] = []
    device = input_ids.device

    with torch.inference_mode():
        _sync_if_cuda(device)
        prefill_t0 = time.perf_counter()
        prefill_out = prefill(model, input_ids, cache=cache)
        _sync_if_cuda(device)
        prefill_ms = (time.perf_counter() - prefill_t0) * 1000.0

        current_logits = prefill_out[:, -1:, :]  # [1, 1, vocab]
        logits.append(current_logits)

        prompt_len = input_ids.shape[1]
        decode_count = 0
        _sync_if_cuda(device)
        decode_t0 = time.perf_counter()
        for step, token_id in enumerate(forced_tokens[:-1], start=1):
            token_tensor = torch.tensor([[token_id]], device=device)
            step_out = decode_step(
                model, token_tensor, position=prompt_len + step - 1, cache=cache
            )
            logits.append(step_out[:, -1:, :])
            decode_count += 1
        _sync_if_cuda(device)
        total_decode_ms = (time.perf_counter() - decode_t0) * 1000.0

    return _ReplayResult(
        logits=torch.cat(logits, dim=1),
        prefill_ms=prefill_ms,
        total_decode_ms=total_decode_ms,
        decode_count=decode_count,
    )


def _memory_by_type(
    cache: object,
    num_layers: int,
    layer_types: tuple[str, ...],
) -> tuple[int, int]:
    sliding = 0
    global_ = 0
    if not hasattr(cache, "memory_bytes"):
        return 0, 0
    for i in range(num_layers):
        b = cache.memory_bytes(i)
        if layer_types[i] == "global":
            global_ += b
        else:
            sliding += b
    return sliding, global_


def _survival_rates(cache: object, num_layers: int) -> dict[int, float]:
    rates: dict[int, float] = {}
    if not hasattr(cache, "seq_length"):
        return rates
    for i in range(num_layers):
        seq_len = cache.seq_length(i)  # type: ignore[union-attr]
        evicted = cache.key_offset(i) if hasattr(cache, "key_offset") else 0
        total_seen = seq_len + evicted
        rates[i] = seq_len / total_seen if total_seen > 0 else 1.0
    return rates


def _cache_pressure(
    cache: object,
    num_layers: int,
    layer_types: tuple[str, ...],
) -> tuple[int, int, float, float, dict[int, int], dict[int, int]]:
    if not hasattr(cache, "seq_length"):
        return 0, 0, 0.0, 0.0, {}, {}

    retained_sliding = 0
    retained_global = 0
    evicted_sliding = 0
    evicted_global = 0
    lengths_by_layer: dict[int, int] = {}
    evictions_by_layer: dict[int, int] = {}

    for i in range(num_layers):
        seq_len = cache.seq_length(i)  # type: ignore[union-attr]
        evicted = cache.key_offset(i) if hasattr(cache, "key_offset") else 0
        lengths_by_layer[i] = seq_len
        evictions_by_layer[i] = evicted
        if layer_types[i] == "global":
            retained_global += seq_len
            evicted_global += evicted
        else:
            retained_sliding += seq_len
            evicted_sliding += evicted

    sliding_seen = retained_sliding + evicted_sliding
    global_seen = retained_global + evicted_global
    sliding_rate = evicted_sliding / sliding_seen if sliding_seen else 0.0
    global_rate = evicted_global / global_seen if global_seen else 0.0

    return (
        evicted_sliding,
        evicted_global,
        sliding_rate,
        global_rate,
        lengths_by_layer,
        evictions_by_layer,
    )


def run_experiment(
    model: GemmaModel,
    tokenizer: GemmaTokenizer,
    config: ExperimentConfig,
) -> list[RunResult]:
    """Run all (strategy, budget, prompt) combinations and return results."""
    results: list[RunResult] = []
    model_config = model.config
    device = next(model.parameters()).device
    dtype_bytes = next(model.parameters()).element_size()

    iso_memory_mode = bool(config.memory_budgets_mb)
    budget_label = "mem_mb" if iso_memory_mode else "budget"
    budget_values: tuple[int, ...] = (
        config.memory_budgets_mb if iso_memory_mode else config.window_budgets
    )

    for local_prompt_idx, prompt_text in enumerate(config.prompts):
        prompt_idx = config.prompt_index_offset + local_prompt_idx
        category = config.categories[local_prompt_idx] if config.categories else "unknown"
        print(
            f"  [{local_prompt_idx + 1}/{len(config.prompts)}] {category}: {prompt_text[:50]!r}",
            flush=True,
        )
        input_ids = torch.tensor([tokenizer.encode(prompt_text)], dtype=torch.long, device=device)
        prompt_tokens = input_ids.shape[1]

        # Reference run with standard strategy
        ref_gen_config = GenerationConfig(
            max_tokens=config.max_tokens,
            temperature=0.0,
            seed=config.seed,
            cache_strategy="standard",
        )
        reference = generate(model, input_ids, ref_gen_config)
        ref_logits: torch.Tensor | None = None
        if reference.step_logits:
            ref_logits = torch.stack(reference.step_logits, dim=1)  # [1, T, vocab]

        for budget in budget_values:
            window_budget = None if iso_memory_mode else budget
            memory_budget_bytes = budget * 1024 * 1024 if iso_memory_mode else None

            for strategy in config.strategies:
                t0 = time.perf_counter()
                cache = create_cache(
                    strategy,
                    model_config,
                    window_budget=window_budget,
                    memory_budget_bytes=memory_budget_bytes,
                    dtype_bytes=dtype_bytes,
                )

                gen_config = GenerationConfig(
                    max_tokens=config.max_tokens,
                    temperature=0.0,
                    seed=config.seed,
                    cache_strategy="standard",  # cache is pre-built; this is ignored
                )
                generate(model, input_ids, gen_config, cache=cache)

                sliding_bytes, global_bytes = _memory_by_type(
                    cache, model_config.num_layers, model_config.layer_types
                )
                survival = _survival_rates(cache, model_config.num_layers)
                (
                    evicted_sliding,
                    evicted_global,
                    eviction_rate_sliding,
                    eviction_rate_global,
                    cache_lengths_by_layer,
                    evictions_by_layer,
                ) = _cache_pressure(cache, model_config.num_layers, model_config.layer_types)

                kl = 0.0
                ppl = 0.0
                prefill_ms = 0.0
                total_decode_ms = 0.0
                decode_ms_per_token = 0.0
                if ref_logits is not None and reference.generated_ids:
                    ref_token_ids = torch.tensor(
                        [reference.generated_ids], dtype=torch.long, device=device
                    )
                    # Always run a teacher-forced replay to capture clean
                    # prefill/decode timings — even for ``standard``, where KL
                    # would be 0 by construction.
                    replay = _replay_forced_tokens(
                        model,
                        input_ids,
                        reference.generated_ids,
                        strategy,
                        window_budget,
                        memory_budget_bytes,
                        dtype_bytes,
                    )
                    t = min(ref_logits.shape[1], replay.logits.shape[1])
                    if strategy != "standard":
                        kl = _kl_divergence(ref_logits[:, :t], replay.logits[:, :t])
                    ppl = perplexity(replay.logits[:, :t], ref_token_ids[:, :t])
                    prefill_ms = replay.prefill_ms
                    total_decode_ms = replay.total_decode_ms
                    decode_ms_per_token = (
                        replay.total_decode_ms / replay.decode_count
                        if replay.decode_count > 0
                        else 0.0
                    )

                elapsed = time.perf_counter() - t0
                print(
                    f"    {strategy:<16} {budget_label}={budget:>4}  "
                    f"ppl={ppl:6.2f}  kl={kl:.4f}  "
                    f"prefill={prefill_ms:.0f}ms decode={decode_ms_per_token:.1f}ms/tok  "
                    f"{elapsed:.1f}s",
                    flush=True,
                )
                results.append(
                    RunResult(
                        strategy=strategy,
                        window_budget=window_budget if window_budget is not None else 0,
                        memory_budget_mb=budget if iso_memory_mode else 0,
                        prompt_idx=prompt_idx,
                        kl_divergence=kl,
                        perplexity=ppl,
                        memory_bytes_sliding=sliding_bytes,
                        memory_bytes_global=global_bytes,
                        token_survival_rates=survival,
                        category=category,
                        prompt_tokens=prompt_tokens,
                        generated_tokens=len(reference.generated_ids),
                        effective_tokens=prompt_tokens + len(reference.generated_ids),
                        evicted_tokens_sliding=evicted_sliding,
                        evicted_tokens_global=evicted_global,
                        eviction_rate_sliding=eviction_rate_sliding,
                        eviction_rate_global=eviction_rate_global,
                        cache_lengths_by_layer=cache_lengths_by_layer,
                        evictions_by_layer=evictions_by_layer,
                        prefill_ms=prefill_ms,
                        total_decode_ms=total_decode_ms,
                        decode_ms_per_token=decode_ms_per_token,
                    )
                )

    return results


def save_results(results: list[RunResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = []
    for r in results:
        d = asdict(r)
        d["token_survival_rates"] = {int(k): v for k, v in d["token_survival_rates"].items()}
        d["cache_lengths_by_layer"] = {int(k): v for k, v in d["cache_lengths_by_layer"].items()}
        d["evictions_by_layer"] = {int(k): v for k, v in d["evictions_by_layer"].items()}
        data.append(d)
    path.write_text(json.dumps(data, indent=2))


def load_results(path: Path) -> list[RunResult]:
    data = json.loads(path.read_text())
    results: list[RunResult] = []
    for d in data:
        row = {
            **d,
            "token_survival_rates": {
                int(k): v for k, v in d.get("token_survival_rates", {}).items()
            },
            "memory_budget_mb": d.get("memory_budget_mb", 0),
            "category": d.get("category", "unknown"),
            "prompt_tokens": d.get("prompt_tokens", 0),
            "generated_tokens": d.get("generated_tokens", 0),
            "effective_tokens": d.get("effective_tokens", 0),
            "evicted_tokens_sliding": d.get("evicted_tokens_sliding", 0),
            "evicted_tokens_global": d.get("evicted_tokens_global", 0),
            "eviction_rate_sliding": d.get("eviction_rate_sliding", 0.0),
            "eviction_rate_global": d.get("eviction_rate_global", 0.0),
            "cache_lengths_by_layer": {
                int(k): v for k, v in d.get("cache_lengths_by_layer", {}).items()
            },
            "evictions_by_layer": {int(k): v for k, v in d.get("evictions_by_layer", {}).items()},
            "prefill_ms": d.get("prefill_ms", 0.0),
            "total_decode_ms": d.get("total_decode_ms", 0.0),
            "decode_ms_per_token": d.get("decode_ms_per_token", 0.0),
        }
        results.append(RunResult(**row))
    return results
