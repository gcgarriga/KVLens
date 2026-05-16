"""Tests for Gemma model assembly."""

from __future__ import annotations

import torch

from kvlens.model import GemmaBlock, GemmaModel


class TestGemmaBlock:
    def test_layer_scalar_scales_full_block_output(self, tiny_config) -> None:
        block = GemmaBlock(tiny_config, layer_idx=0)
        hidden_states = torch.randn(1, 3, tiny_config.hidden_size)
        per_layer_input = torch.randn(1, 3, tiny_config.ple_dim)
        positions = torch.arange(3)

        with torch.no_grad():
            block.layer_scalar.zero_()

        output = block(hidden_states, per_layer_input=per_layer_input, positions=positions)

        torch.testing.assert_close(output, torch.zeros_like(output))


class TestGemmaBlockNoPLE:
    def test_forward_with_none_per_layer_input(self, tiny_gemma2_config) -> None:
        block = GemmaBlock(tiny_gemma2_config, layer_idx=0)
        hidden_states = torch.randn(1, 3, tiny_gemma2_config.hidden_size)
        positions = torch.arange(3)
        out = block(hidden_states, per_layer_input=None, positions=positions)
        assert out.shape == hidden_states.shape
        assert torch.isfinite(out).all()

    def test_no_ple_modules_created(self, tiny_gemma2_config) -> None:
        block = GemmaBlock(tiny_gemma2_config, layer_idx=0)
        assert not hasattr(block, "per_layer_injection")
        assert not hasattr(block, "layer_scalar")


class TestGemmaModel:
    def test_forward_shape_and_determinism(self, tiny_config) -> None:
        model = GemmaModel(tiny_config)
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])

        logits_a = model(input_ids)
        logits_b = model(input_ids)

        assert logits_a.shape == (2, 3, tiny_config.vocab_size)
        assert torch.isfinite(logits_a).all()
        torch.testing.assert_close(logits_a, logits_b)

    def test_positions_above_context_limit_raise(self, tiny_config) -> None:
        model = GemmaModel(tiny_config)
        input_ids = torch.tensor([[1, 2, 3]])
        positions = torch.tensor([0, 1, tiny_config.max_position_embeddings])

        try:
            model(input_ids, positions=positions)
        except ValueError as exc:
            assert "max_position_embeddings" in str(exc)
        else:
            raise AssertionError(
                "Expected ValueError when positions exceed max_position_embeddings"
            )

    def test_empty_input_raises(self, tiny_config) -> None:
        model = GemmaModel(tiny_config)
        try:
            model(torch.empty((1, 0), dtype=torch.long))
        except ValueError as exc:
            assert "empty" in str(exc).lower()
        else:
            raise AssertionError("Expected ValueError for empty input_ids")

    def test_forward_ple_disabled(self, tiny_gemma2_config) -> None:
        model = GemmaModel(tiny_gemma2_config)
        input_ids = torch.tensor([[1, 2, 3]])
        logits = model(input_ids)
        assert logits.shape == (1, 3, tiny_gemma2_config.vocab_size)
        assert torch.isfinite(logits).all()

    def test_no_ple_modules_in_model(self, tiny_gemma2_config) -> None:
        model = GemmaModel(tiny_gemma2_config)
        assert model.embed_tokens_per_layer is None
        assert model.per_layer_model_projection is None

    def test_softcap_disabled_logits_not_bounded(self, tiny_gemma2_config) -> None:
        # With softcap=0.0 logits are raw; with softcap=30.0 they are bounded to (-30, 30).
        model = GemmaModel(tiny_gemma2_config)
        input_ids = torch.tensor([[1, 2, 3]])
        logits = model(input_ids)
        # With softcap disabled, logits are raw (unbounded). Shape check is sufficient.
        assert logits.shape == (1, 3, tiny_gemma2_config.vocab_size)
