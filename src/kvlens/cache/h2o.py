"""H2O (Heavy Hitter Oracle) KV cache: evict by lowest accumulated attention score."""

from __future__ import annotations

import torch

from kvlens.cache.hybrid import HybridCache
from kvlens.config import Gemma4Config


class H2OCache(HybridCache):
    """Evicts the cached token with the lowest cumulative attention score.

    Scores are fed in via update_scores() called from attention.py after
    each decode step (q_len == 1). Without scores, falls back to FIFO.
    Global layers (window_size=None) are never evicted.
    """

    def __init__(self, config: Gemma4Config) -> None:
        super().__init__(config)
        self.scores: dict[int, torch.Tensor] = {}  # layer_idx → [seq]
        self._token_positions: dict[int, torch.Tensor] = {}

    def update_scores(self, layer_idx: int, attention_weights: torch.Tensor) -> None:
        """Accumulate per-token attention scores. attention_weights: [B, H, 1, kv_len]."""
        incoming = attention_weights.detach().mean(dim=(0, 1, 2))  # [kv_len]
        existing = self.scores.get(layer_idx)
        if existing is not None and existing.shape[0] == incoming.shape[0]:
            self.scores[layer_idx] = existing + incoming
        else:
            self.scores[layer_idx] = incoming

    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int,
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

        window_size = self.config.layer_params(layer_idx).window_size
        evicted = 0
        if window_size is not None and combined_key.shape[-2] > window_size:
            n_new = key.shape[-2]
            seq_total = combined_key.shape[-2]
            evicted = seq_total - window_size

            existing_scores = self.scores.get(layer_idx)
            if existing_scores is not None:
                # Initialise new-token scores at the mean of existing scores
                new_scores = existing_scores.mean().expand(n_new).to(existing_scores.device)
                all_scores = torch.cat([existing_scores, new_scores])
                _, keep_idx = all_scores.topk(window_size, largest=True, sorted=True)
                keep_idx = keep_idx.sort().values  # preserve temporal order
                key_idx = keep_idx.to(combined_key.device)
                combined_key = combined_key[:, :, key_idx, :]
                combined_value = combined_value[:, :, key_idx, :]
                combined_pos = combined_pos[keep_idx.to(combined_pos.device)]
                self.scores[layer_idx] = all_scores[keep_idx]
            else:
                # No scores yet: fall back to FIFO
                combined_key = combined_key[:, :, -window_size:, :]
                combined_value = combined_value[:, :, -window_size:, :]
                combined_pos = combined_pos[-window_size:]

        self.storage[layer_idx] = (combined_key, combined_value)
        self._token_positions[layer_idx] = combined_pos
        self.last_evictions[layer_idx] = evicted
        self._cumulative_evictions[layer_idx] = (
            self._cumulative_evictions.get(layer_idx, 0) + evicted
        )
        return combined_key, combined_value

    def cache_positions(self, layer_idx: int) -> torch.Tensor | None:
        return self._token_positions.get(layer_idx)

    def reset(self) -> None:
        super().reset()
        self.scores.clear()
        self._token_positions.clear()
