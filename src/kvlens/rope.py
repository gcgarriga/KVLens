"""Rotary Position Embeddings for Gemma-style attention."""

from __future__ import annotations

import torch
from torch import nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension by swapping and negating halves."""
    half_dim = x.shape[-1] // 2
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]
    return torch.cat((-x2, x1), dim=-1)


class RoPE(nn.Module):
    """Rotary Position Embeddings with optional partial rotation."""

    def __init__(
        self,
        head_dim: int,
        theta: float = 10000.0,
        partial_factor: float = 1.0,
    ) -> None:
        super().__init__()
        if head_dim < 2 or head_dim % 2 != 0:
            raise ValueError(f"head_dim must be an even integer >= 2, got {head_dim}")
        if theta <= 0.0:
            raise ValueError(f"theta must be > 0.0, got {theta}")
        if not 0.0 < partial_factor <= 1.0:
            raise ValueError(f"partial_factor must be in (0.0, 1.0], got {partial_factor}")

        rotary_dim = int(head_dim * partial_factor)
        if rotary_dim < 2 or rotary_dim % 2 != 0:
            raise ValueError(
                "head_dim * partial_factor must produce an even rotary dimension >= 2, "
                f"got {rotary_dim}"
            )

        self.head_dim = head_dim
        self.theta = theta
        self.partial_factor = partial_factor
        self.rotary_dim = rotary_dim

        inv_freq = 1.0 / (
            theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        if x.shape[-1] != self.head_dim:
            raise ValueError(
                f"x last dimension ({x.shape[-1]}) must match head_dim ({self.head_dim})"
            )

        seq_len = x.shape[-2]
        if positions is None:
            positions = torch.arange(seq_len, device=x.device)
        elif positions.ndim != 1 or positions.shape[0] != seq_len:
            raise ValueError(
                "positions must be a 1D tensor with length matching x.shape[-2], "
                f"got shape {tuple(positions.shape)} for seq_len={seq_len}"
            )

        rotary_part = x[..., : self.rotary_dim]
        passthrough = x[..., self.rotary_dim :]

        angles = torch.outer(positions.to(device=self.inv_freq.device).float(), self.inv_freq)  # type: ignore[arg-type]  # torch register_buffer types inv_freq as Tensor|Module
        embeddings = torch.cat((angles, angles), dim=-1)

        view_shape = [1] * (x.ndim - 2) + [seq_len, self.rotary_dim]
        cos = embeddings.cos().to(device=x.device, dtype=x.dtype).view(*view_shape)
        sin = embeddings.sin().to(device=x.device, dtype=x.dtype).view(*view_shape)

        rotated = (rotary_part * cos) + (rotate_half(rotary_part) * sin)
        if not passthrough.numel():
            return rotated
        return torch.cat((rotated, passthrough), dim=-1)
