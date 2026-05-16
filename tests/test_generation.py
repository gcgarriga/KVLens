"""Tests for generation and sampling integration."""

from __future__ import annotations

import torch

from kvlens.cache.hybrid import HybridCache
from kvlens.cache.layertype import LayerTypeAwareCache
from kvlens.cache.pyramidkv import PyramidKVCache
from kvlens.config import GenerationConfig
from kvlens.generation import generate, prefill, teacher_forced_logits_with_cache
from kvlens.instrument import MetricsCollector
from kvlens.model import GemmaModel


class TestGeneration:
    def test_prefill_decode_matches_full_forward(self, tiny_config) -> None:
        model = GemmaModel(tiny_config)
        input_ids = torch.tensor([[1, 2, 3, 4]])

        full_logits = model(input_ids[:, :-1])
        cached_logits = teacher_forced_logits_with_cache(
            model,
            input_ids,
            cache=HybridCache(tiny_config),
        )

        torch.testing.assert_close(cached_logits, full_logits, atol=1e-5, rtol=1e-5)

    def test_generation_is_deterministic_with_seed(self, tiny_config) -> None:
        model = GemmaModel(tiny_config)
        input_ids = torch.tensor([[1, 2, 3]])
        config = GenerationConfig(max_tokens=4, temperature=0.8, top_k=5, top_p=0.9, seed=7)

        result_a = generate(model, input_ids, config)
        result_b = generate(model, input_ids, config)

        assert result_a.generated_ids == result_b.generated_ids
        torch.testing.assert_close(result_a.sequences, result_b.sequences)

    def test_naive_and_standard_generation_match_for_greedy(self, tiny_config) -> None:
        model = GemmaModel(tiny_config)
        input_ids = torch.tensor([[1, 2, 3]])
        naive = GenerationConfig(max_tokens=4, cache_strategy="naive", temperature=0.0)
        standard = GenerationConfig(max_tokens=4, cache_strategy="standard", temperature=0.0)

        naive_result = generate(model, input_ids, naive)
        standard_result = generate(model, input_ids, standard)

        assert naive_result.generated_ids == standard_result.generated_ids

    def test_step_callback_sees_newly_appended_token(self, tiny_config) -> None:
        """Regression: callback must fire after sequences is extended."""
        model = GemmaModel(tiny_config)
        input_ids = torch.tensor([[1, 2, 3]])
        prompt_len = input_ids.shape[1]
        seen: list[int] = []

        def cb(sequences, _metrics) -> None:
            seen.append(sequences.shape[1])

        generate(
            model,
            input_ids,
            GenerationConfig(max_tokens=3, temperature=0.0),
            step_callback=cb,
        )

        assert seen == [prompt_len + 1, prompt_len + 2, prompt_len + 3]

    def test_generate_outputs_have_no_grad(self, tiny_config) -> None:
        model = GemmaModel(tiny_config)
        input_ids = torch.tensor([[1, 2, 3]])
        result = generate(model, input_ids, GenerationConfig(max_tokens=2, temperature=0.0))
        for logits in result.step_logits:
            assert not logits.requires_grad

    def test_generate_first_step_logits_match_no_cache_prefill_above_window(
        self, tiny_config
    ) -> None:
        torch.manual_seed(123)
        model = GemmaModel(tiny_config).eval()
        input_ids = (torch.arange(80).view(1, 80) % tiny_config.vocab_size).long()

        with torch.inference_mode():
            expected = prefill(model, input_ids, cache=None)[:, -1, :]
            result = generate(
                model,
                input_ids,
                GenerationConfig(max_tokens=1, temperature=0.0),
            )

        torch.testing.assert_close(result.step_logits[0], expected, atol=1e-5, rtol=1e-5)

    def test_generate_with_ple_disabled(self, tiny_gemma2_config) -> None:
        model = GemmaModel(tiny_gemma2_config)
        input_ids = torch.tensor([[1, 2, 3]])
        result = generate(model, input_ids, GenerationConfig(max_tokens=4, temperature=0.0))
        assert len(result.generated_ids) > 0
        assert result.sequences.shape[1] > input_ids.shape[1]

    def test_generate_ple_disabled_deterministic(self, tiny_gemma2_config) -> None:
        model = GemmaModel(tiny_gemma2_config)
        input_ids = torch.tensor([[1, 2, 3]])
        cfg = GenerationConfig(max_tokens=4, temperature=0.0, seed=42)
        assert (
            generate(model, input_ids, cfg).generated_ids
            == generate(model, input_ids, cfg).generated_ids
        )

    def test_generate_records_cache_metrics_for_pyramidkv(self, tiny_config) -> None:
        model = GemmaModel(tiny_config)
        metrics = MetricsCollector(enabled=True)

        generate(
            model,
            torch.tensor([[1, 2, 3]]),
            GenerationConfig(max_tokens=1, temperature=0.0),
            metrics=metrics,
            cache=PyramidKVCache(tiny_config, min_budget=8),
        )

        assert metrics.steps[0]["cache"]
        assert 0 in metrics.steps[0]["cache"]

    def test_generate_records_cache_metrics_for_layertype(self, tiny_config) -> None:
        model = GemmaModel(tiny_config)
        metrics = MetricsCollector(enabled=True)

        generate(
            model,
            torch.tensor([[1, 2, 3]]),
            GenerationConfig(max_tokens=1, temperature=0.0),
            metrics=metrics,
            cache=LayerTypeAwareCache(
                tiny_config, sliding_strategy="h2o", global_strategy="standard"
            ),
        )

        assert metrics.steps[0]["cache"]
        assert 0 in metrics.steps[0]["cache"]
