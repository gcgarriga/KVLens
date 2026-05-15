"""Low-overhead instrumentation helpers."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import Any

import torch


class MetricsCollector:
    """Collect generation metrics only when explicitly enabled."""

    def __init__(self, enabled: bool = False, on_step: Callable[..., Any] | None = None) -> None:
        self.enabled = enabled
        self.on_step = on_step
        self.steps: list[dict[str, object]] = []
        self._current_step: dict[str, object] | None = None

    def start_step(self, step_index: int) -> None:
        if not self.enabled:
            return
        self._current_step = {
            "step": step_index,
            "cache": {},
            "attention": {},
            "latency_ms": {},
        }

    def record_cache(self, layer_idx: int, bytes_used: int, evicted: int = 0) -> None:
        if not self.enabled or self._current_step is None:
            return
        cache = self._current_step["cache"]
        if not isinstance(cache, dict):
            return
        cache[layer_idx] = {
            "bytes": int(bytes_used),
            "evicted": int(evicted),
        }

    def record_attention(
        self,
        layer_idx: int,
        attention_weights: torch.Tensor,
        cached_tokens: int,
        new_tokens: int,
    ) -> None:
        if not self.enabled or self._current_step is None:
            return

        probs = attention_weights.detach().float()
        if cached_tokens > 0:
            old_mass = float(probs[..., :cached_tokens].sum(dim=-1).mean().item())
        else:
            old_mass = 0.0
        new_mass = float(
            probs[..., cached_tokens : cached_tokens + new_tokens].sum(dim=-1).mean().item()
        )
        entropy = float((-(probs * probs.clamp_min(1e-9).log()).sum(dim=-1)).mean().item())

        attention = self._current_step["attention"]
        if not isinstance(attention, dict):
            return
        attention[layer_idx] = {
            "cached_mass": old_mass,
            "new_mass": new_mass,
            "entropy": entropy,
        }

    def record_latency(self, name: str, milliseconds: float) -> None:
        if not self.enabled or self._current_step is None:
            return
        latencies = self._current_step["latency_ms"]
        if not isinstance(latencies, dict):
            return
        latencies[name] = float(milliseconds)

    def finalize_step(self, generated_token: int | None = None) -> None:
        if not self.enabled or self._current_step is None:
            return
        self._current_step["generated_token"] = generated_token
        step_data = deepcopy(self._current_step)
        self.steps.append(step_data)
        if self.on_step is not None:
            self.on_step(deepcopy(self._current_step))
        self._current_step = None

    def latest(self) -> dict[str, object] | None:
        if self._current_step is not None:
            return self._current_step
        if self.steps:
            return self.steps[-1]
        return None

    def latency_history(self, name: str) -> list[float]:
        values: list[float] = []
        for step in self.steps:
            latency_ms = step.get("latency_ms", {})
            if isinstance(latency_ms, dict) and name in latency_ms:
                values.append(float(latency_ms[name]))
        return values
