"""Per-layer embedding components used by Gemma 4."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from kvlens.norm import RMSNorm


class PerLayerEmbedder(nn.Module):
    """Token lookup for per-layer inputs."""

    def __init__(self, vocab_size: int, num_layers: int, ple_dim: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.ple_dim = ple_dim
        if ple_dim > 0:
            self.embedding = nn.Embedding(vocab_size, num_layers * ple_dim)
            self.register_buffer("scale", torch.tensor(math.sqrt(ple_dim)), persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor | None:
        if self.ple_dim == 0:
            return None
        embeddings = self.embedding(input_ids) * self.scale
        return embeddings.view(*input_ids.shape, self.num_layers, self.ple_dim)


class PerLayerProjection(nn.Module):
    """Projection from model hidden states into per-layer inputs."""

    def __init__(self, hidden_size: int, num_layers: int, ple_dim: int, eps: float) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.ple_dim = ple_dim
        self.proj = nn.Linear(hidden_size, num_layers * ple_dim, bias=False)
        self.norm = RMSNorm(ple_dim, eps=eps)
        self.scale = hidden_size**-0.5
        self.combine_scale = 2.0**-0.5

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        projected = self.proj(hidden_states) * self.scale
        projected = projected.view(*hidden_states.shape[:-1], self.num_layers, self.ple_dim)
        return self.norm(projected)

    def combine(
        self,
        hidden_states: torch.Tensor,
        token_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        projected = self.forward(hidden_states)
        if token_embeddings is None:
            return projected
        return (projected + token_embeddings) * self.combine_scale


class PerLayerInjection(nn.Module):
    """Inject one per-layer input into a decoder block."""

    def __init__(self, hidden_size: int, ple_dim: int, eps: float) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, ple_dim, bias=False)
        self.proj = nn.Linear(ple_dim, hidden_size, bias=False)
        self.norm = RMSNorm(hidden_size, eps=eps)

    def forward(self, hidden_states: torch.Tensor, per_layer_input: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        gated = F.gelu(self.gate_proj(hidden_states), approximate="tanh")
        output = self.proj(gated * per_layer_input)
        output = self.norm(output)
        return residual + output
