"""Normalization layers used by KVLens."""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, hidden_size: int, eps: float = 1e-6, with_scale: bool = True) -> None:
        super().__init__()
        if hidden_size < 1:
            raise ValueError(f"hidden_size must be >= 1, got {hidden_size}")
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0.0, got {eps}")

        self.hidden_size = hidden_size
        self.eps = eps

        if with_scale:
            # Gemma stores norm weights as deltas from 1; effective scale = (1 + weight).
            # Initialized to zeros so the identity is (1 + 0) = 1 at construction.
            self.weight = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.register_parameter("weight", None)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        mean_squared = x.pow(2).mean(dim=-1, keepdim=True) + self.eps
        return x * torch.pow(mean_squared, -0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        output = self._norm(x.float())
        if self.weight is not None:
            output = output * (1.0 + self.weight.float())
        return output.to(input_dtype)
