"""Tests for experiment metrics."""

from __future__ import annotations

import math

import torch

from kvlens.experiments.metrics import perplexity, token_survival_rate


def test_perplexity_perfect_prediction() -> None:
    vocab = 10
    seq = 5
    logits = torch.full((1, seq, vocab), -1e9)
    token_ids = torch.arange(seq).unsqueeze(0)
    for i in range(seq):
        logits[0, i, i] = 0.0

    ppl = perplexity(logits, token_ids)
    assert abs(ppl - 1.0) < 0.01


def test_perplexity_uniform_distribution() -> None:
    vocab = 100
    seq = 4
    logits = torch.zeros(1, seq, vocab)
    token_ids = torch.zeros(1, seq, dtype=torch.long)

    ppl = perplexity(logits, token_ids)
    expected = math.exp(math.log(vocab))
    assert abs(ppl - expected) < 1.0


def test_token_survival_rate_no_eviction() -> None:
    seq_lengths = [10, 10, 10, 10]
    evictions = [0, 0, 0, 0]
    rate = token_survival_rate(seq_lengths, evictions)
    assert all(abs(r - 1.0) < 1e-6 for r in rate)


def test_token_survival_rate_with_eviction() -> None:
    seq_lengths = [10, 10, 10]
    evictions = [0, 1, 1]
    rate = token_survival_rate(seq_lengths, evictions)
    assert abs(rate[0] - 1.0) < 1e-6
    assert abs(rate[1] - 0.9) < 1e-5
    assert abs(rate[2] - 0.9) < 1e-5


def test_token_survival_rate_zero_seq_length() -> None:
    rate = token_survival_rate([0, 10], [0, 0])
    assert abs(rate[0] - 1.0) < 1e-6
    assert abs(rate[1] - 1.0) < 1e-6


def test_perplexity_all_positions_contribute() -> None:
    vocab = 10
    seq = 4
    logits = torch.full((1, seq, vocab), -1e9)
    token_ids = torch.zeros(1, seq, dtype=torch.long)
    logits[0, 0, 0] = 0.0
    for i in range(1, seq):
        logits[0, i] = torch.zeros(vocab)

    ppl = perplexity(logits, token_ids)
    assert ppl > 1.0
    assert ppl < float(vocab)


def test_perplexity_batch_averaged() -> None:
    vocab = 10
    seq = 3
    batch = 2
    logits = torch.zeros(batch, seq, vocab)
    token_ids = torch.zeros(batch, seq, dtype=torch.long)

    ppl = perplexity(logits, token_ids)
    assert abs(ppl - float(vocab)) < 1.0
