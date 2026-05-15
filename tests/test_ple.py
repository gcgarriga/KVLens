"""Tests for per-layer embedding components."""

from __future__ import annotations

import torch

from kvlens.ple import PerLayerEmbedder, PerLayerInjection, PerLayerProjection


class TestPerLayerEmbedder:
    def test_output_shape(self) -> None:
        embedder = PerLayerEmbedder(vocab_size=16, num_layers=4, ple_dim=8)
        input_ids = torch.tensor([[1, 2, 3]])

        output = embedder(input_ids)

        assert output.shape == (1, 3, 4, 8)


class TestPerLayerProjection:
    def test_projection_and_combination_shapes(self) -> None:
        projection = PerLayerProjection(hidden_size=6, num_layers=4, ple_dim=8, eps=1e-6)
        hidden_states = torch.randn(2, 3, 6)
        token_embeddings = torch.randn(2, 3, 4, 8)

        projected = projection(hidden_states)
        combined = projection.combine(hidden_states, token_embeddings)

        assert projected.shape == (2, 3, 4, 8)
        assert combined.shape == (2, 3, 4, 8)

    def test_different_layers_get_different_values(self) -> None:
        projection = PerLayerProjection(hidden_size=4, num_layers=3, ple_dim=2, eps=1e-6)
        hidden_states = torch.randn(1, 2, 4)

        output = projection(hidden_states)

        assert not torch.equal(output[:, :, 0, :], output[:, :, 1, :])


class TestPerLayerInjection:
    def test_gate_and_residual_shape(self) -> None:
        injection = PerLayerInjection(hidden_size=6, ple_dim=4, eps=1e-6)
        hidden_states = torch.randn(2, 3, 6)
        per_layer_input = torch.randn(2, 3, 4)

        output = injection(hidden_states, per_layer_input)
        gate = injection.gate_proj(hidden_states)

        assert output.shape == hidden_states.shape
        assert gate.shape == per_layer_input.shape
