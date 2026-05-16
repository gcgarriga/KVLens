"""Experiment quality metrics: perplexity and token survival rate."""

from __future__ import annotations

import math

import torch


def perplexity(logits: torch.Tensor, token_ids: torch.Tensor) -> float:
    """Compute token-level perplexity of token_ids under logits.

    Args:
        logits: [batch, seq, vocab] — logits at each position.
        token_ids: [batch, seq] — the ground-truth token at each position.

    Returns:
        Scalar perplexity (exp of mean negative log-prob).
    """
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=token_ids.unsqueeze(-1)).squeeze(
        -1
    )  # [batch, seq]
    mean_neg_log_prob = -token_log_probs.mean().item()
    return math.exp(mean_neg_log_prob)


def token_survival_rate(
    seq_lengths: list[int],
    evictions: list[int],
) -> list[float]:
    """Compute fraction of cached tokens that survive each decode step.

    Args:
        seq_lengths: cache seq_length before each decode step.
        evictions: number of tokens evicted at each step (0 if within window).

    Returns:
        List of survival rates in [0, 1], one per step.
    """
    rates = []
    for seq_len, evicted in zip(seq_lengths, evictions, strict=True):
        if seq_len == 0:
            rates.append(1.0)
        else:
            rates.append((seq_len - evicted) / seq_len)
    return rates
