"""StreamingLLM KV cache: keep attention sinks + recency window."""

from __future__ import annotations

import torch

from kvlens.cache.hybrid import HybridCache
from kvlens.config import Gemma4Config


class StreamingLLMCache(HybridCache):
    """Keeps first sink_tokens positions always; fills rest with most-recent tokens.

    Global layers (window_size=None) are unaffected — they keep all tokens.
    Exposes cache_positions() for attention mask building so sink tokens
    are not incorrectly blocked by the sliding window mask.
    """

    def __init__(self, config: Gemma4Config, sink_tokens: int = 4) -> None:
        super().__init__(config)
        self.sink_tokens = sink_tokens
        self._token_positions: dict[int, torch.Tensor] = {}

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
            combined_pos = positions if positions is not None else torch.arange(key.shape[-2])
        else:
            combined_key = torch.cat((cached[0], key), dim=-2)
            combined_value = torch.cat((cached[1], value), dim=-2)
            prev_pos = self._token_positions.get(layer_idx, torch.arange(cached[0].shape[-2]))
            if positions is not None:
                new_pos = positions
            else:
                new_pos = torch.arange(
                    prev_pos[-1].item() + 1,
                    prev_pos[-1].item() + 1 + key.shape[-2],
                )
            combined_pos = torch.cat([prev_pos.to(new_pos.device), new_pos])

        window_size = self.config.layer_params(layer_idx).window_size
        evicted = 0
        if window_size is not None and combined_key.shape[-2] > window_size:
            evicted = combined_key.shape[-2] - window_size
            sink_count = min(self.sink_tokens, window_size)
            recent_size = max(0, window_size - sink_count)
            sink_key = combined_key[:, :, :sink_count, :]
            sink_val = combined_value[:, :, :sink_count, :]
            sink_pos = combined_pos[:sink_count]
            if recent_size > 0:
                recent_key = combined_key[:, :, -recent_size:, :]
                recent_val = combined_value[:, :, -recent_size:, :]
                combined_key = torch.cat([sink_key, recent_key], dim=-2)
                combined_value = torch.cat([sink_val, recent_val], dim=-2)
                combined_pos = torch.cat([sink_pos, combined_pos[-recent_size:]])
            else:
                combined_key = sink_key
                combined_value = sink_val
                combined_pos = sink_pos

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
        self._token_positions.clear()
