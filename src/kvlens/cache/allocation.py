"""Per-layer cache budget allocation for iso-memory experiments.

Given a total cache budget in bytes and a strategy, compute the per-layer token
cap that each strategy should use. This lets every strategy compete on equal
total memory rather than equal per-layer windows.

Sliding-layer caps never exceed the model's native sliding-window size: tokens
retained beyond that window are invisible to the trained sliding attention
pattern. When a finite all-layer/proportional allocation has surplus bytes after
that cap, the surplus is reassigned uniformly to finite global-layer caps; this
keeps those bytes on layers that can attend to them, even for pyramid variants.

Allocation policies by strategy family:

- ``standard``, ``naive``, ``quantized``: returns empty dict (no cap to apply).
- ``h2o`` / ``snapkv`` / ``streaming`` (hybrid-adapted): globals stay full
  (window_size=None override); total budget is split equally across sliding
  layers.
- ``pyramidkv`` (hybrid-adapted): globals stay full; sliding layers receive a
  linear pyramid distribution (shallowest gets most, deepest least) summing to
  the total budget.
- ``all_layers_h2o`` / ``all_layers_snapkv`` / ``all_layers_streaming``: total
  budget is split equally across every layer.
- ``all_layers_pyramidkv``: linear pyramid across all layers (no type
  distinction).
- ``proportional_h2o`` / ``proportional_snapkv`` / ``proportional_streaming``:
  iso-token across every layer — each layer caches the same number of tokens,
  so byte share scales with per-token cost. On homogeneous-head_dim models
  this is identical to ``all_layers_*``; on heterogeneous models (Gemma 4)
  larger-head_dim layers receive a proportionally larger byte budget.
- ``proportional_pyramidkv``: linear pyramid in *tokens* across all layers
  (shallowest gets most tokens, regardless of head_dim).
- ``hybrid_h2o`` / ``hybrid_snapkv`` / ``hybrid_streaming``: globals receive the
  entire memory budget split uniformly; sliding layers keep their architectural
  window (no per-layer override emitted) so they evict via ``HybridCache``'s
  FIFO rule when the prompt exceeds the trained window.
"""

from __future__ import annotations

from kvlens.config import Gemma4Config


def per_layer_token_bytes(config: Gemma4Config, layer_idx: int, dtype_bytes: int) -> int:
    """Bytes per cached token for one layer (counts both K and V)."""
    head_dim = config.layer_params(layer_idx).head_dim
    return head_dim * config.num_kv_heads * 2 * dtype_bytes


def _bytes_to_tokens(layer_bytes: int, per_token: int) -> int:
    """Convert a byte budget to a token count, with a minimum of one token."""
    return max(1, layer_bytes // per_token)


def _pyramid_weights(n: int) -> list[int]:
    """Linear pyramid: layer rank r ∈ [0, n-1] gets weight (n - r). Shallowest = n, deepest = 1."""
    return [n - r for r in range(n)]


def _allocate_uniform(
    config: Gemma4Config, indices: list[int], total_bytes: int, dtype_bytes: int
) -> dict[int, int | None]:
    if not indices:
        return {}
    per_layer = total_bytes // len(indices)
    return {
        i: _bytes_to_tokens(per_layer, per_layer_token_bytes(config, i, dtype_bytes))
        for i in indices
    }


def _allocate_pyramid(
    config: Gemma4Config, indices: list[int], total_bytes: int, dtype_bytes: int
) -> dict[int, int | None]:
    if not indices:
        return {}
    weights = _pyramid_weights(len(indices))
    total_weight = sum(weights)
    out: dict[int, int | None] = {}
    for rank, layer_idx in enumerate(indices):
        layer_bytes = weights[rank] * total_bytes // total_weight
        out[layer_idx] = _bytes_to_tokens(
            layer_bytes, per_layer_token_bytes(config, layer_idx, dtype_bytes)
        )
    return out


def _allocate_proportional(
    config: Gemma4Config, indices: list[int], total_bytes: int, dtype_bytes: int
) -> dict[int, int | None]:
    """Iso-token allocation: every layer gets the same token count.

    Bytes per layer scale with per-token cost; on uniform-head_dim models this
    collapses exactly to ``_allocate_uniform``.
    """
    if not indices:
        return {}
    total_per_token = sum(per_layer_token_bytes(config, i, dtype_bytes) for i in indices)
    tokens_per_layer = max(1, total_bytes // total_per_token)
    return {i: tokens_per_layer for i in indices}


def _allocate_proportional_pyramid(
    config: Gemma4Config, indices: list[int], total_bytes: int, dtype_bytes: int
) -> dict[int, int | None]:
    """Linear pyramid in tokens across ``indices``: shallowest gets the most
    tokens; deepest the fewest. Total bytes consumed ≤ ``total_bytes``."""
    if not indices:
        return {}
    weights = _pyramid_weights(len(indices))
    weighted_per_token = sum(
        weights[r] * per_layer_token_bytes(config, indices[r], dtype_bytes)
        for r in range(len(indices))
    )
    base_tokens = max(1, total_bytes // weighted_per_token)
    return {indices[r]: max(1, weights[r] * base_tokens) for r in range(len(indices))}


def _allocate_to_globals_only(
    config: Gemma4Config, total_bytes: int, dtype_bytes: int
) -> dict[int, int | None]:
    """For hybrid_* strategies: globals get the full memory budget, slidings keep
    their architectural window (no override emitted, so the config retains its
    per-layer ``window_size``).
    """
    global_idx = [i for i in range(config.num_layers) if config.layer_types[i] == "global"]
    return _allocate_uniform(config, global_idx, total_bytes, dtype_bytes)


def _cap_sliding_budgets(
    config: Gemma4Config,
    budgets: dict[int, int | None],
    dtype_bytes: int,
) -> dict[int, int | None]:
    out = dict(budgets)
    surplus_bytes = 0
    finite_global_indices: list[int] = []

    for layer_idx, layer_type in enumerate(config.layer_types):
        budget = out.get(layer_idx)
        if layer_type == "global":
            if isinstance(budget, int):
                finite_global_indices.append(layer_idx)
            continue
        window_size = config.layer_params(layer_idx).window_size
        if window_size is not None and isinstance(budget, int) and budget > window_size:
            surplus_bytes += (budget - window_size) * per_layer_token_bytes(
                config, layer_idx, dtype_bytes
            )
            out[layer_idx] = window_size

    if surplus_bytes and finite_global_indices:
        per_global_surplus = surplus_bytes // len(finite_global_indices)
        for layer_idx in finite_global_indices:
            budget = out[layer_idx]
            if isinstance(budget, int):
                out[layer_idx] = budget + (
                    per_global_surplus // per_layer_token_bytes(config, layer_idx, dtype_bytes)
                )

    return out


def allocate_memory_budget(
    strategy: str,
    config: Gemma4Config,
    total_bytes: int,
    dtype_bytes: int = 2,
) -> dict[int, int | None]:
    """Per-layer token cap for ``strategy`` at a total cache budget of ``total_bytes``.

    Globals receiving an explicit ``None`` keep full context; layers absent from
    the returned dict are unaffected by the budget (used for ``standard`` /
    ``naive`` / ``quantized``).
    """
    if total_bytes <= 0:
        raise ValueError(f"total_bytes must be positive, got {total_bytes}")

    n = config.num_layers
    sliding_idx = [i for i in range(n) if config.layer_types[i] == "sliding"]
    global_idx = [i for i in range(n) if config.layer_types[i] == "global"]
    all_idx = list(range(n))

    if strategy in ("standard", "naive", "quantized"):
        return {}

    if strategy in ("h2o", "snapkv", "streaming"):
        budgets: dict[int, int | None] = {i: None for i in global_idx}
        budgets.update(_allocate_uniform(config, sliding_idx, total_bytes, dtype_bytes))
        return _cap_sliding_budgets(config, budgets, dtype_bytes)

    if strategy == "pyramidkv":
        budgets = {i: None for i in global_idx}
        budgets.update(_allocate_pyramid(config, sliding_idx, total_bytes, dtype_bytes))
        return _cap_sliding_budgets(config, budgets, dtype_bytes)

    if strategy in ("all_layers_h2o", "all_layers_snapkv", "all_layers_streaming"):
        return _cap_sliding_budgets(
            config, _allocate_uniform(config, all_idx, total_bytes, dtype_bytes), dtype_bytes
        )

    if strategy == "all_layers_pyramidkv":
        return _cap_sliding_budgets(
            config, _allocate_pyramid(config, all_idx, total_bytes, dtype_bytes), dtype_bytes
        )

    if strategy in ("proportional_h2o", "proportional_snapkv", "proportional_streaming"):
        return _cap_sliding_budgets(
            config, _allocate_proportional(config, all_idx, total_bytes, dtype_bytes), dtype_bytes
        )

    if strategy == "proportional_pyramidkv":
        return _cap_sliding_budgets(
            config,
            _allocate_proportional_pyramid(config, all_idx, total_bytes, dtype_bytes),
            dtype_bytes,
        )

    if strategy in ("hybrid_h2o", "hybrid_snapkv", "hybrid_streaming"):
        return _allocate_to_globals_only(config, total_bytes, dtype_bytes)

    raise ValueError(f"unknown strategy for memory budget allocation: '{strategy}'")
