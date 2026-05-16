"""Sampling helpers for autoregressive generation."""

from __future__ import annotations

import torch


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0.0:
        return logits
    return logits / temperature


def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0 or top_k >= logits.shape[-1]:
        return logits

    values, _ = torch.topk(logits, top_k, dim=-1)
    threshold = values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    remove_mask = cumulative > top_p
    remove_mask[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(remove_mask, float("-inf"))

    unsorted = torch.full_like(logits, float("-inf"))
    unsorted.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
    return unsorted


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if temperature == 0.0:
        return logits.argmax(dim=-1)

    filtered = apply_temperature(logits, temperature)
    filtered = apply_top_k(filtered, top_k)
    filtered = apply_top_p(filtered, top_p)
    probs = torch.softmax(filtered, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
