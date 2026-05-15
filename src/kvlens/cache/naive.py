"""Baseline cache that stores nothing and recomputes everything."""

from __future__ import annotations

import torch


class NaiveCache:
    """No-op cache used by the full recomputation baseline."""

    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int,
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del layer_idx, positions
        return key, value

    def get(self, layer_idx: int) -> None:
        del layer_idx
        return None

    def reset(self) -> None:
        return None

    def seq_length(self, layer_idx: int) -> int:
        del layer_idx
        return 0

    def key_offset(self, layer_idx: int) -> int:
        del layer_idx
        return 0

    def memory_bytes(self, layer_idx: int) -> int:
        del layer_idx
        return 0
