"""Tests for RMSNorm and Rotary Position Embeddings."""

from __future__ import annotations

import math

import torch

from kvlens.norm import RMSNorm
from kvlens.rope import RoPE


class TestRMSNorm:
    def test_shape_and_norm(self) -> None:
        norm = RMSNorm(hidden_size=4, eps=1e-6)
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

        output = norm(x)
        expected = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)

        assert output.shape == x.shape
        torch.testing.assert_close(output, expected)

    def test_without_scale_has_no_learnable_parameters(self) -> None:
        norm = RMSNorm(hidden_size=4, with_scale=False)

        assert norm.weight is None
        assert list(norm.parameters()) == []


class TestRoPE:
    def test_full_rotation_matches_hand_computed_values(self) -> None:
        rope = RoPE(head_dim=4, theta=10000.0, partial_factor=1.0)
        x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]]])

        output = rope(x)

        expected = torch.tensor(
            [
                [
                    [
                        [1.0, 2.0, 3.0, 4.0],
                        [
                            5.0 * math.cos(1.0) - 7.0 * math.sin(1.0),
                            6.0 * math.cos(0.01) - 8.0 * math.sin(0.01),
                            7.0 * math.cos(1.0) + 5.0 * math.sin(1.0),
                            8.0 * math.cos(0.01) + 6.0 * math.sin(0.01),
                        ],
                    ]
                ]
            ],
            dtype=torch.float32,
        )

        torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-6)

    def test_partial_rotation_leaves_tail_dims_unchanged(self) -> None:
        rope = RoPE(head_dim=8, theta=1_000_000.0, partial_factor=0.25)
        x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]]])
        positions = torch.tensor([1])

        output = rope(x, positions=positions)

        torch.testing.assert_close(output[..., 2:], x[..., 2:])
        assert not torch.equal(output[..., :2], x[..., :2])

    def test_custom_positions_offset_rotation(self) -> None:
        rope = RoPE(head_dim=4, theta=10000.0)
        x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
        positions = torch.tensor([2])

        output = rope(x, positions=positions)
        expected = torch.tensor(
            [
                [
                    [
                        [
                            1.0 * math.cos(2.0) - 3.0 * math.sin(2.0),
                            2.0 * math.cos(0.02) - 4.0 * math.sin(0.02),
                            3.0 * math.cos(2.0) + 1.0 * math.sin(2.0),
                            4.0 * math.cos(0.02) + 2.0 * math.sin(0.02),
                        ]
                    ]
                ]
            ],
            dtype=torch.float32,
        )

        torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-6)
