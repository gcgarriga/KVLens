"""Experiment replay regressions for prefill/cache semantics."""

from __future__ import annotations

import torch

from kvlens.config import Gemma4Config, LayerParams, RoPEConfig
from kvlens.experiments.runner import _replay_forced_tokens
from kvlens.generation import prefill
from kvlens.model import GemmaModel


def _config() -> Gemma4Config:
    return Gemma4Config(
        num_layers=6,
        hidden_size=128,
        intermediate_size=256,
        num_heads=4,
        num_kv_heads=2,
        vocab_size=512,
        max_position_embeddings=256,
        layer_types=("sliding", "sliding", "global", "sliding", "sliding", "global"),
        sliding=LayerParams(head_dim=32, rope=RoPEConfig(), window_size=16),
        global_=LayerParams(
            head_dim=32,
            rope=RoPEConfig(theta=1_000_000.0),
            window_size=None,
        ),
        ple_dim=0,
        num_kv_shared_layers=0,
    )


def test_replay_first_logit_matches_no_cache_prefill_last_logit() -> None:
    torch.manual_seed(123)
    config = _config()
    model = GemmaModel(config).eval()
    input_ids = (torch.arange(40).view(1, 40) % config.vocab_size).long()
    forced_tokens = [1, 2, 3]

    with torch.inference_mode():
        expected = prefill(model, input_ids, cache=None)[:, -1:, :]
        replay = _replay_forced_tokens(
            model,
            input_ids,
            forced_tokens,
            strategy="standard",
            window_budget=None,
            memory_budget_bytes=None,
            dtype_bytes=4,
        )

    torch.testing.assert_close(replay.logits[:, :1, :], expected, atol=1e-5, rtol=1e-5)
