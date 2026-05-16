"""Explicit attention implementation for Gemma 4."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
from torch import nn

from kvlens.cache import CacheProtocol
from kvlens.config import Gemma4Config
from kvlens.norm import RMSNorm
from kvlens.rope import RoPE


@dataclass(frozen=True)
class AttentionStats:
    """Small summary used by instrumentation and tests."""

    cached_tokens: int
    new_tokens: int


class Attention(nn.Module):
    """Single attention module parameterized per layer."""

    def __init__(self, config: Gemma4Config, layer_idx: int) -> None:
        super().__init__()
        params = config.layer_params(layer_idx)

        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.gqa_ratio = config.gqa_ratio
        self.head_dim = params.head_dim
        self.window_size = params.window_size

        self.is_kv_shared = config.is_kv_shared_layer(layer_idx)
        self.kv_shared_source = config.kv_shared_source(layer_idx)
        self.stores_kv = config.stores_kv_for_sharing(layer_idx)

        self._qk_norm = config.qk_norm
        self._attn_scale = 1.0 / math.sqrt(self.head_dim)
        self._attn_softcap = config.attn_logit_softcapping

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        if config.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Shared layers reuse K/V from a non-shared source — no K/V weights needed
        if not self.is_kv_shared:
            kv_dim = self.num_kv_heads * self.head_dim
            self.k_proj = nn.Linear(config.hidden_size, kv_dim, bias=False)
            self.v_proj = nn.Linear(config.hidden_size, kv_dim, bias=False)
            if config.qk_norm:
                self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
                self.v_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, with_scale=False)

        self.rope = RoPE(
            head_dim=self.head_dim,
            theta=params.rope.theta,
            partial_factor=params.rope.partial_factor,
        )

    def _reshape_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, num_heads, self.head_dim).transpose(1, 2)

    def _expand_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_heads == self.num_kv_heads:
            return x
        return x.repeat_interleave(self.gqa_ratio, dim=1)

    def _build_attention_mask(
        self,
        positions: torch.Tensor,
        key_len: int,
        key_offset: int = 0,
        cached_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if positions.ndim != 1:
            raise ValueError(f"positions must be 1D, got shape {tuple(positions.shape)}")

        query_positions = positions.view(1, 1, -1, 1)

        if cached_positions is not None:
            # Cache manages sparse retention; explicit positions preserve sinks.
            key_positions = cached_positions.to(positions.device).view(1, 1, 1, -1)
            return key_positions <= query_positions

        key_positions = torch.arange(
            key_offset, key_offset + key_len, device=positions.device
        ).view(1, 1, 1, -1)
        mask = key_positions <= query_positions

        if self.window_size is not None:
            min_positions = query_positions - self.window_size + 1
            mask = mask & (key_positions >= min_positions)

        return mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor | None = None,
        cache: CacheProtocol | None = None,
        metrics: object | None = None,
        shared_kv_states: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        if positions is None:
            positions = torch.arange(seq_len, device=hidden_states.device)
        if positions.ndim != 1 or positions.shape[0] != seq_len:
            raise ValueError(
                "positions must be a 1D tensor with length matching the sequence, "
                f"got shape {tuple(positions.shape)} for seq_len={seq_len}"
            )

        query_states = self._reshape_heads(self.q_proj(hidden_states), self.num_heads)
        if self._qk_norm:
            query_states = self.q_norm(query_states)
        query_states = self.rope(query_states, positions=positions)

        cached_tokens = 0

        # Shared layers reuse K/V from their source layer (no projection)
        cache_layer_idx = self.layer_idx
        if self.is_kv_shared and shared_kv_states is not None:
            assert self.kv_shared_source is not None  # guaranteed when is_kv_shared
            cache_layer_idx = self.kv_shared_source
            key_states, value_states = shared_kv_states[self.kv_shared_source]
        else:
            key_states, value_states, cached_tokens = self._compute_kv(
                hidden_states, positions, cache, metrics
            )
        defer_cache_update = cache is not None and not self.is_kv_shared and seq_len > 1
        update_key_states = key_states
        update_value_states = value_states
        if defer_cache_update:
            assert cache is not None  # narrowed by defer_cache_update
            cached = cache.get(self.layer_idx)
            if cached is not None:
                key_states = torch.cat((cached[0], key_states), dim=-2)
                value_states = torch.cat((cached[1], value_states), dim=-2)

        # Store K/V for sharing with downstream layers
        if self.stores_kv and shared_kv_states is not None:
            shared_kv_states[self.layer_idx] = (key_states, value_states)

        expanded_keys = self._expand_kv(key_states)
        expanded_values = self._expand_kv(value_states)

        cached_positions = None
        key_len = expanded_keys.shape[-2]
        use_cache_metadata = key_len != seq_len
        if cache is not None and use_cache_metadata and hasattr(cache, "cache_positions"):
            cached_positions = cache.cache_positions(cache_layer_idx)
            if (
                cached_positions is not None
                and cached_positions.shape[0] != key_len
                and cached_positions.shape[0] + positions.shape[0] == key_len
            ):
                cached_positions = torch.cat([cached_positions.to(positions.device), positions])

        key_offset = 0
        if cache is not None and use_cache_metadata:
            key_offset = cache.key_offset(cache_layer_idx)

        scores = torch.matmul(query_states, expanded_keys.transpose(-2, -1))
        if not self._qk_norm:
            scores = scores * self._attn_scale
        if self._attn_softcap > 0.0:
            scores = self._attn_softcap * torch.tanh(scores / self._attn_softcap)
        mask = self._build_attention_mask(
            positions, expanded_keys.shape[-2], key_offset, cached_positions
        )
        scores = scores.masked_fill(~mask, -torch.finfo(scores.dtype).max)
        # Upcast to float32 for softmax precision (matches HF)
        attention_weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(scores.dtype)

        # SnapKV hook: accumulate attention scores during prefill (q_len > 1)
        if (
            cache is not None
            and hasattr(cache, "accumulate_prefill_scores")
            and attention_weights.shape[-2] > 1
        ):
            cache.accumulate_prefill_scores(self.layer_idx, attention_weights)

        # H2O hook: accumulate attention scores during decode only (q_len == 1)
        if (
            cache is not None
            and hasattr(cache, "update_scores")
            and attention_weights.shape[-2] == 1
        ):
            cache.update_scores(self.layer_idx, attention_weights)

        if defer_cache_update:
            assert cache is not None  # narrowed by defer_cache_update
            cache_start = time.perf_counter()
            cache.update(update_key_states, update_value_states, self.layer_idx, positions)
            if metrics is not None and hasattr(metrics, "record_latency"):
                metrics.record_latency(
                    "cache_update", (time.perf_counter() - cache_start) * 1000.0
                )

        output = torch.matmul(attention_weights, expanded_values)
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                seq_len,
                self.num_heads * self.head_dim,
            )
        )
        output = self.o_proj(output)

        if metrics is not None and hasattr(metrics, "record_attention"):
            metrics.record_attention(
                layer_idx=self.layer_idx,
                attention_weights=attention_weights,
                cached_tokens=cached_tokens,
                new_tokens=seq_len,
            )

        return output

    def _compute_kv(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        cache: CacheProtocol | None,
        metrics: object | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Compute K/V from projections, optionally updating cache."""
        key_states = self._reshape_heads(self.k_proj(hidden_states), self.num_kv_heads)
        if self._qk_norm:
            key_states = self.k_norm(key_states)
        key_states = self.rope(key_states, positions=positions)

        value_states = self._reshape_heads(self.v_proj(hidden_states), self.num_kv_heads)
        if self._qk_norm:
            value_states = self.v_norm(value_states)

        cached_tokens = 0
        if cache is not None:
            cached_tokens = cache.seq_length(self.layer_idx)
            if key_states.shape[-2] == 1:
                cache_start = time.perf_counter()
                key_states, value_states = cache.update(
                    key_states, value_states, self.layer_idx, positions
                )
                if metrics is not None and hasattr(metrics, "record_latency"):
                    metrics.record_latency(
                        "cache_update", (time.perf_counter() - cache_start) * 1000.0
                    )

        return key_states, value_states, cached_tokens
