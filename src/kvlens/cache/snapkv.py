"""SnapKV: one-shot token selection at prefill end based on attention scores."""

from __future__ import annotations

import torch

from kvlens.cache.hybrid import HybridCache
from kvlens.config import Gemma4Config


class SnapKVCache(HybridCache):
    """Selects important tokens once at the prefill→decode transition.

    During prefill: stores all tokens, accumulates attention scores.
    On first decode step: trims cache to top-window_size tokens by score.
    After that: FIFO append within remaining budget.
    Global layers (window_size=None) are never evicted.
    """

    def __init__(self, config: Gemma4Config, observation_window: int = 32) -> None:
        super().__init__(config)
        self.observation_window = observation_window
        self._prefill_scores: dict[int, torch.Tensor] = {}  # layer → [kv_seq]
        self._snapped: set[int] = set()
        self._token_positions: dict[int, torch.Tensor] = {}

    def accumulate_prefill_scores(self, layer_idx: int, attention_weights: torch.Tensor) -> None:
        """Called from attention.py during prefill (q_len > 1).

        attention_weights: [B, H, q_len, kv_len]
        Averages over last observation_window query positions.
        """
        obs = min(self.observation_window, attention_weights.shape[-2])
        scores = attention_weights[:, :, -obs:, :].mean(dim=(0, 1, 2))  # [kv_len]
        existing = self._prefill_scores.get(layer_idx)
        if existing is not None and existing.shape == scores.shape:
            self._prefill_scores[layer_idx] = existing + scores
        else:
            self._prefill_scores[layer_idx] = scores

    def _snap(self, layer_idx: int) -> None:
        """Trim cache to window_size tokens by prefill score. Called once."""
        cached = self.storage.get(layer_idx)
        if cached is None:
            self._snapped.add(layer_idx)
            return

        window_size = self.config.layer_params(layer_idx).window_size
        if window_size is None or cached[0].shape[-2] <= window_size:
            self._snapped.add(layer_idx)
            return

        k, v = cached
        scores = self._prefill_scores.get(layer_idx)
        if scores is not None and scores.shape[0] == k.shape[-2]:
            keep_idx = scores.topk(window_size, largest=True).indices.sort().values
        else:
            # No scores: FIFO — keep most recent window_size tokens
            keep_idx = torch.arange(
                k.shape[-2] - window_size,
                k.shape[-2],
                device=k.device,
            )

        evicted = k.shape[-2] - window_size
        key_idx = keep_idx.to(k.device)
        positions = self._token_positions.get(
            layer_idx,
            torch.arange(k.shape[-2], device=k.device),
        )
        self.storage[layer_idx] = (k[:, :, key_idx, :], v[:, :, key_idx, :])
        self._token_positions[layer_idx] = positions[keep_idx.to(positions.device)]
        self.last_evictions[layer_idx] = evicted
        self._cumulative_evictions[layer_idx] = (
            self._cumulative_evictions.get(layer_idx, 0) + evicted
        )
        self._snapped.add(layer_idx)

    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int,
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        window_size = self.config.layer_params(layer_idx).window_size
        snap_evicted = 0

        # Snap on first decode step (key.shape[-2] == 1, not yet snapped)
        if key.shape[-2] == 1 and layer_idx not in self._snapped:
            self._snap(layer_idx)
            snap_evicted = self.last_evictions.get(layer_idx, 0)

        cached = self.storage.get(layer_idx)
        if cached is None:
            combined_key = key
            combined_value = value
            combined_pos = (
                positions
                if positions is not None
                else torch.arange(key.shape[-2], device=key.device)
            )
        else:
            combined_key = torch.cat((cached[0], key), dim=-2)
            combined_value = torch.cat((cached[1], value), dim=-2)
            prev_pos = self._token_positions.get(
                layer_idx,
                torch.arange(cached[0].shape[-2], device=cached[0].device),
            )
            new_pos = (
                positions
                if positions is not None
                else torch.arange(
                    prev_pos[-1].item() + 1,
                    prev_pos[-1].item() + 1 + key.shape[-2],
                    device=prev_pos.device,
                )
            )
            combined_pos = torch.cat([prev_pos.to(new_pos.device), new_pos])

        evicted = 0
        if (
            layer_idx in self._snapped
            and window_size is not None
            and combined_key.shape[-2] > window_size
        ):
            evicted = combined_key.shape[-2] - window_size
            combined_key = combined_key[:, :, -window_size:, :]
            combined_value = combined_value[:, :, -window_size:, :]
            combined_pos = combined_pos[-window_size:]

        self.storage[layer_idx] = (combined_key, combined_value)
        self._token_positions[layer_idx] = combined_pos
        self.last_evictions[layer_idx] = snap_evicted + evicted
        self._cumulative_evictions[layer_idx] = (
            self._cumulative_evictions.get(layer_idx, 0) + evicted
        )
        return combined_key, combined_value

    def cache_positions(self, layer_idx: int) -> torch.Tensor | None:
        return self._token_positions.get(layer_idx)

    def reset(self) -> None:
        super().reset()
        self._prefill_scores.clear()
        self._snapped.clear()
        self._token_positions.clear()
