"""Autoregressive generation loop with explicit prefill and decode."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, TypeGuard

import torch

from kvlens.cache import CacheProtocol, HybridCache, NaiveCache, create_cache
from kvlens.config import GenerationConfig
from kvlens.model import GemmaModel
from kvlens.sampling import sample_next_token


@dataclass
class GenerationResult:
    """Return type for generation."""

    sequences: torch.Tensor
    generated_ids: list[int]
    step_logits: list[torch.Tensor]


class CacheMetricsProtocol(CacheProtocol, Protocol):
    config: object
    last_evictions: dict[int, int]


def prefill(
    model: GemmaModel,
    input_ids: torch.Tensor,
    cache: CacheProtocol | None = None,
    metrics: object | None = None,
    position_offset: int = 0,
) -> torch.Tensor:
    """Process the full prompt in a single forward pass, returning logits."""
    positions = torch.arange(
        position_offset,
        position_offset + input_ids.shape[1],
        device=input_ids.device,
    )
    return model(input_ids, positions=positions, cache=cache, metrics=metrics)


def decode_step(
    model: GemmaModel,
    token_ids: torch.Tensor,
    *,
    position: int,
    cache: CacheProtocol,
    metrics: object | None = None,
) -> torch.Tensor:
    """Run a single decode step at the given position, returning logits."""
    positions = torch.tensor([position], device=token_ids.device)
    return model(token_ids, positions=positions, cache=cache, metrics=metrics)


def teacher_forced_logits_with_cache(
    model: GemmaModel,
    input_ids: torch.Tensor,
    cache: HybridCache,
    metrics: object | None = None,
) -> torch.Tensor:
    """Validate cache by comparing teacher-forced logits against full forward."""
    if input_ids.shape[1] < 2:
        raise ValueError("teacher forcing requires at least 2 tokens")

    cache.reset()
    logits: list[torch.Tensor] = []

    prefill_logits = prefill(model, input_ids[:, :1], cache=cache, metrics=metrics)
    logits.append(prefill_logits[:, -1, :])

    for position in range(1, input_ids.shape[1] - 1):
        step_logits = decode_step(
            model,
            input_ids[:, position : position + 1],
            position=position,
            cache=cache,
            metrics=metrics,
        )
        logits.append(step_logits[:, -1, :])

    return torch.stack(logits, dim=1)


def generate(
    model: GemmaModel,
    input_ids: torch.Tensor,
    generation_config: GenerationConfig,
    metrics: object | None = None,
    step_callback: Callable[..., Any] | None = None,
    cache: CacheProtocol | None = None,
    position_offset: int = 0,
    stop_token_ids: set[int] | None = None,
) -> GenerationResult:
    """Run autoregressive generation: prefill the prompt, then decode token by token.

    Args:
        cache: If provided, reuse this cache (for multi-turn). Otherwise create a new one.
        position_offset: Starting position for RoPE when continuing from an existing cache.
        stop_token_ids: Token IDs that terminate generation (default: {1} i.e. EOS).
    """
    with torch.inference_mode():
        return _generate_impl(
            model,
            input_ids,
            generation_config,
            metrics=metrics,
            step_callback=step_callback,
            cache=cache,
            position_offset=position_offset,
            stop_token_ids=stop_token_ids,
        )


def _generate_impl(
    model: GemmaModel,
    input_ids: torch.Tensor,
    generation_config: GenerationConfig,
    metrics: object | None = None,
    step_callback: Callable[..., Any] | None = None,
    cache: CacheProtocol | None = None,
    position_offset: int = 0,
    stop_token_ids: set[int] | None = None,
) -> GenerationResult:
    if input_ids.shape[0] != 1:
        raise ValueError("generate currently supports only batch size 1")

    if stop_token_ids is None:
        stop_token_ids = {1}  # EOS token

    generator = None
    if generation_config.seed is not None:
        generator = torch.Generator(device=input_ids.device)
        generator.manual_seed(generation_config.seed)

    if cache is None:
        cache = create_cache(generation_config.cache_strategy, model.config)
    sequences = input_ids.clone()
    generated_ids: list[int] = []
    step_logits: list[torch.Tensor] = []

    if metrics is not None and hasattr(metrics, "start_step"):
        metrics.start_step(0)

    if isinstance(cache, NaiveCache):
        forward_start = time.perf_counter()
        logits = model(sequences, metrics=metrics)
        current_logits = logits[:, -1, :]
        if metrics is not None and hasattr(metrics, "record_latency"):
            metrics.record_latency("forward", (time.perf_counter() - forward_start) * 1000.0)
    else:
        forward_start = time.perf_counter()
        logits = prefill(
            model,
            sequences,
            cache=cache,
            metrics=metrics,
            position_offset=position_offset,
        )
        current_logits = logits[:, -1, :]
        if metrics is not None and hasattr(metrics, "record_latency"):
            metrics.record_latency("forward", (time.perf_counter() - forward_start) * 1000.0)
            if _can_record_cache_state(cache):
                _record_cache_state(cache, metrics)

    next_position = position_offset + sequences.shape[1]

    for _step in range(generation_config.max_tokens):
        sample_start = time.perf_counter()
        next_token = sample_next_token(
            current_logits,
            temperature=generation_config.temperature,
            top_k=generation_config.top_k,
            top_p=generation_config.top_p,
            generator=generator,
        )
        if metrics is not None and hasattr(metrics, "record_latency"):
            metrics.record_latency("sampling", (time.perf_counter() - sample_start) * 1000.0)

        step_logits.append(current_logits.detach().clone())
        token_id = int(next_token.item())
        generated_ids.append(token_id)
        if metrics is not None and hasattr(metrics, "finalize_step"):
            metrics.finalize_step(generated_token=token_id)

        sequences = torch.cat((sequences, next_token[:, None]), dim=1)

        if step_callback is not None:
            step_callback(sequences, metrics)

        if token_id in stop_token_ids:
            break

        if metrics is not None and hasattr(metrics, "start_step"):
            metrics.start_step(len(generated_ids))

        if isinstance(cache, NaiveCache):
            forward_start = time.perf_counter()
            logits = model(sequences, metrics=metrics)
            current_logits = logits[:, -1, :]
            if metrics is not None and hasattr(metrics, "record_latency"):
                metrics.record_latency("forward", (time.perf_counter() - forward_start) * 1000.0)
        else:
            forward_start = time.perf_counter()
            logits = decode_step(
                model,
                next_token[:, None],
                position=next_position,
                cache=cache,
                metrics=metrics,
            )
            current_logits = logits[:, -1, :]
            next_position += 1
            if metrics is not None and hasattr(metrics, "record_latency"):
                metrics.record_latency("forward", (time.perf_counter() - forward_start) * 1000.0)
                if _can_record_cache_state(cache):
                    _record_cache_state(cache, metrics)

    return GenerationResult(
        sequences=sequences,
        generated_ids=generated_ids,
        step_logits=step_logits,
    )


def _can_record_cache_state(cache: object) -> TypeGuard[CacheMetricsProtocol]:
    return isinstance(cache, CacheProtocol) and all(
        hasattr(cache, attr) for attr in ("config", "last_evictions")
    )


def _record_cache_state(cache: CacheMetricsProtocol, metrics: object) -> None:
    if not hasattr(metrics, "record_cache"):
        return
    if not hasattr(cache.config, "num_layers"):
        return
    for layer_idx in range(cache.config.num_layers):
        metrics.record_cache(
            layer_idx=layer_idx,
            bytes_used=cache.memory_bytes(layer_idx),
            evicted=cache.last_evictions.get(layer_idx, 0),
        )
