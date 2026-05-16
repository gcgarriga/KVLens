"""PyramidKV: depth-based budget allocation — deeper layers get less cache."""

from __future__ import annotations

import torch

from kvlens.config import Gemma4Config


class PyramidKVCache:
    """Allocates KV budgets that decrease linearly with sliding layer depth.

    Shallowest sliding layer: window_size tokens.
    Deepest sliding layer: min_budget tokens.
    Global layers: no eviction (keep full context).

    Depth-based baseline for comparing against type-based allocation (LayerTypeAware).
    """

    def __init__(
        self,
        config: Gemma4Config,
        min_budget: int | None = None,
        apply_to_all_layers: bool = False,
        precomputed_budgets: dict[int, int | None] | None = None,
    ) -> None:
        self.config = config
        self.min_budget = min_budget
        # apply_to_all_layers: pyramid across every layer (canonical PyramidKV port).
        # Default False: pyramid across sliding layers only, globals keep full context.
        self.apply_to_all_layers = apply_to_all_layers
        # precomputed_budgets: when provided (iso-memory mode), use these per-layer
        # caps directly and skip internal pyramid computation. The allocator
        # already produced the desired distribution.
        if precomputed_budgets is not None:
            self._budgets = dict(precomputed_budgets)
        else:
            self._budgets = self._compute_budgets()
        self._storage: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.last_evictions: dict[int, int] = {}
        self._cumulative_evictions: dict[int, int] = {}

    def _compute_budgets(self) -> dict[int, int | None]:
        budgets: dict[int, int | None] = {}
        if self.apply_to_all_layers:
            target_indices = list(range(self.config.num_layers))
        else:
            target_indices = [i for i, t in enumerate(self.config.layer_types) if t == "sliding"]
        n = len(target_indices)
        for rank, layer_idx in enumerate(target_indices):
            window = self.config.layer_params(layer_idx).window_size
            if window is None:
                budgets[layer_idx] = None
            elif n == 1:
                budgets[layer_idx] = window
            else:
                min_budget = self.min_budget
                if min_budget is None:
                    min_budget = max(1, window // 4)
                min_budget = min(min_budget, window)
                fraction = rank / (n - 1)  # 0 = shallowest, 1 = deepest
                budget = int(window * (1 - fraction) + min_budget * fraction)
                budgets[layer_idx] = max(min_budget, budget)
        if not self.apply_to_all_layers:
            for layer_idx, t in enumerate(self.config.layer_types):
                if t == "global":
                    budgets[layer_idx] = None
        return budgets

    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int,
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del positions
        cached = self._storage.get(layer_idx)
        if cached is None:
            combined_key = key
            combined_value = value
        else:
            combined_key = torch.cat((cached[0], key), dim=-2)
            combined_value = torch.cat((cached[1], value), dim=-2)

        budget = self._budgets.get(layer_idx)
        evicted = 0
        if budget is not None and combined_key.shape[-2] > budget:
            evicted = combined_key.shape[-2] - budget
            combined_key = combined_key[:, :, -budget:, :]
            combined_value = combined_value[:, :, -budget:, :]

        self._storage[layer_idx] = (combined_key, combined_value)
        self.last_evictions[layer_idx] = evicted
        self._cumulative_evictions[layer_idx] = (
            self._cumulative_evictions.get(layer_idx, 0) + evicted
        )
        return combined_key, combined_value

    def get(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        return self._storage.get(layer_idx)

    def reset(self) -> None:
        self._storage.clear()
        self.last_evictions.clear()
        self._cumulative_evictions.clear()

    def seq_length(self, layer_idx: int) -> int:
        cached = self._storage.get(layer_idx)
        return cached[0].shape[-2] if cached else 0

    def key_offset(self, layer_idx: int) -> int:
        return self._cumulative_evictions.get(layer_idx, 0)

    def memory_bytes(self, layer_idx: int) -> int:
        cached = self._storage.get(layer_idx)
        if cached is None:
            return 0
        k, v = cached
        return (k.nelement() + v.nelement()) * k.element_size()
