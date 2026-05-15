# KVLens Architecture

## Overview

KVLens is a from-scratch PyTorch implementation of the Gemma 4 E4B text decoder. Every component is explicit — there are no library-level attention abstractions, no hidden caching, no framework magic. The codebase is structured so each file has one responsibility.

## Module dependency graph

```
Leaf modules (no kvlens imports):
  config.py, norm.py, rope.py, sampling.py, tokenizer.py, cache/naive.py

Level 1:
  attention.py        → config, norm, rope
  ple.py              → norm
  cache/hybrid.py     → config
  cache/allocation.py → config

Level 2:
  cache/quantized.py  → config, cache/hybrid
  cache/streaming.py  → config, cache/hybrid
  cache/h2o.py        → config, cache/hybrid
  cache/layertype.py  → config, cache/hybrid, cache/h2o
  cache/snapkv.py     → config, cache/hybrid
  cache/pyramidkv.py  → config, cache/hybrid
  cache/__init__.py   → all cache modules + cache/allocation
  model.py            → attention, config, norm, ple

Level 3:
  generation.py       → cache, config, model, sampling
  instrument.py       → (standalone, only torch)
  weights.py          → model

Level 4:
  experiments/runner.py → cache, config, generation, model, tokenizer
  experiments/plot.py   → experiments/runner
  session.py            → cache, config, model, tokenizer

Level 5:
  cli.py → config, experiments, generation, instrument, model, tokenizer, weights
```

## Gemma 4 E4B block equation

Each of the 42 decoder blocks follows this exact order:

```
1. h = input_layernorm(x)
   h = attention(h, cache)
   h = post_attention_layernorm(h)
   x = x + h                           # attention residual

2. h = pre_feedforward_layernorm(x)
   h = ffn(h)                          # gated MLP: up + gate with GELU, then down
   h = post_feedforward_layernorm(h)
   x = x + h                           # FFN residual

3. gate = GELU(linear(x))              # hidden_size → ple_dim
   gated = gate * per_layer_input[i]   # element-wise
   proj = linear(gated)                # ple_dim → hidden_size
   x = x + RMSNorm(proj)              # PLE residual

4. x = x * layer_scalar                # learnable per-layer scalar
```

## Hybrid attention

The model has two layer types with different configurations:

| Property | Sliding (35 layers) | Global (7 layers) |
|----------|--------------------|--------------------|
| Pattern | Layers 0-4, 6-10, ... | Every 6th: 5, 11, 17, ... |
| head_dim | 256 | 512 |
| RoPE θ | 10,000 | 1,000,000 |
| RoPE partial_factor | 1.0 (full rotation) | 0.25 (128 of 512 dims) |
| Window | 512 tokens | None (full context) |

A single `Attention` class handles both, parameterized by `LayerParams` from `config.layer_params(layer_idx)`.

**GQA** is native: 8 query heads, 2 KV heads (4:1 ratio). K and V are broadcast across query groups during attention computation.

**Q/K norms** replace traditional 1/√d_k scaling. Both Q and K have RMSNorm after projection (before RoPE). V has RMSNorm without learnable scale. Attention scaling is 1.0.

## Per-Layer Embeddings (PLE)

PLE is a two-stage process:

**Stage 1 — Preprocessing (once per forward pass, at model level):**
- Single embedding table `[vocab_size, num_layers × ple_dim]` → lookup → reshape to `[batch, seq, num_layers, ple_dim]`, scaled by √ple_dim.
- Linear projection from hidden states → `[batch, seq, num_layers, ple_dim]`, scaled by 1/√hidden_size, then RMSNorm.
- Sum the two, scale by 1/√2. This produces `per_layer_inputs`.

**Stage 2 — Per-layer injection (in each block, after attention + FFN):**
- GELU gate: `hidden_size → ple_dim`
- Element-wise multiply with `per_layer_inputs[:, :, layer_idx]`
- Project back: `ple_dim → hidden_size`
- RMSNorm → residual add

PLE accounts for ~2.8B of the model's ~8B parameters (~5.6 GB in bfloat16).

## Cache strategies

All strategies implement the same duck-typed interface:

```python
update(key, value, layer_idx, positions=None) -> (cached_key, cached_value)
get(layer_idx) -> tuple[Tensor, Tensor] | None
reset()
seq_length(layer_idx) -> int
key_offset(layer_idx) -> int
memory_bytes(layer_idx) -> int
```

`HybridCache` additionally exposes `last_evictions: dict[int, int]` for instrumentation.

Optional attention hooks (called from `attention.py` when present):
- `cache_positions(layer_idx)` — position indices for masked attention (StreamingLLM)
- `accumulate_prefill_scores(layer_idx, scores)` — stores prefill attention weights (SnapKV)
- `update_scores(layer_idx, scores)` — per-decode score accumulation (H2O)

The factory in `cache/__init__.py` dispatches two families:

**Hybrid-adapted** (default for new experiments) — read `config.layer_params(idx).window_size` and skip eviction on global layers (where it is `None`). Practical for hybrid attention models.

| Strategy | Behavior |
|----------|----------|
| `standard` | FIFO sliding window + full global context. The reference. |
| `streaming` | Keep first N sink tokens + recency window (StreamingLLM). |
| `h2o` | Evict by lowest accumulated attention score (H2O). |
| `snapkv` | One-shot token selection at prefill end; zero decode overhead. |
| `pyramidkv` | Depth-based budget across sliding layers. |

**Canonical baselines** — uniform per-layer eviction across *all* layers, including globals. Reproduces the original paper formulations and serves as the "naive port" baseline that fails on hybrid models.

| Strategy | Behavior |
|----------|----------|
| `all_layers_h2o` | Canonical Zhang et al. H2O — uniform eviction across all 26/42 layers. |
| `all_layers_snapkv` | SnapKV applied uniformly. |
| `all_layers_streaming` | StreamingLLM applied uniformly. |
| `all_layers_pyramidkv` | Depth-pyramid across all layers (not just sliding). |

Other strategies: `naive` (no storage, full recomputation), `quantized` (int8 compression — orthogonal axis to selection, not part of the eviction sweep).

`LayerTypeAwareCache` (in `cache/layertype.py`) is retained as an explicit construction primitive but is no longer surfaced via the factory — the hybrid-adapted strategies already implement that pattern, so the factory branch was redundant.

Note: KV sharing (last 18 layers reuse K/V from earlier same-type layers, ~43% memory savings) is part of the model architecture itself and is applied regardless of cache strategy — see `model.forward()`'s `shared_kv_states` dict. KV sharing is Gemma 4 only; `gemma2_2b()` sets `num_kv_shared_layers=0`.

Cache tensors are always `[batch, kv_heads, seq, head_dim]`, handling heterogeneous head dims (256 for sliding, 512 for global layers in Gemma 4).

## Budget allocation

`create_cache(strategy, config, *, window_budget=None, memory_budget_bytes=None, dtype_bytes=2)` exposes two mutually-exclusive budget modes:

- **Iso-per-layer** (`window_budget`) — every sliding layer gets the same token cap. Used for early per-layer budget sweeps. Globals are unbounded for hybrid-adapted strategies.
- **Iso-memory** (`memory_budget_bytes`) — total cache budget in bytes; each strategy allocates across layers per its own policy via `cache/allocation.py::allocate_memory_budget`. This is the primary public-study axis because it lets canonical baselines and type-aware variants compete on equal total memory.

| Strategy family | Iso-memory allocation policy |
|---|---|
| `all_layers_*` | Uniform: total / num_layers per layer |
| `h2o` / `snapkv` / `streaming` (hybrid) | Globals stay full; total / num_sliding per sliding layer |
| `pyramidkv` (hybrid) | Pyramid across sliding only; globals full |
| `all_layers_pyramidkv` | Pyramid across all layers |

Per-layer caps are applied via `Gemma4Config.with_per_layer_budgets(budgets)`, which returns a frozen-copy config where `layer_params(idx)` returns a per-layer-overridden `LayerParams`. PyramidKV bypasses its internal pyramid in iso-memory mode by accepting a `precomputed_budgets` dict directly.

## Generation pipeline

```
generate(model, input_ids, config)
  ├── prefill(model, input_ids, cache)     # process full prompt
  │     → logits, updated cache
  └── loop:
        decode_step(model, next_token, cache, position)
          → logits, updated cache
        sample_next_token(logits, config)
          → token_id
        stop if token_id == EOS or position >= max_tokens
```

**Prefill** processes the entire prompt in one forward pass.
**Decode** processes one token at a time, using the cache.

## Instrumentation

`MetricsCollector` records per-step metrics when enabled:
- Cache memory per layer (bytes, differentiating sliding vs global)
- Sliding window eviction events
- Attention mass distribution (cached vs new tokens)
- Attention entropy per head
- Wall-clock latency (forward pass, cache update, sampling)

When `enabled=False`, no metrics are recorded, no tensors retained. Only a trivial `if` check on the hot path.

The batch experiment runner uses a separate, simpler instrumentation path: it wraps `prefill()` and the decode loop with `time.perf_counter()` and `torch.cuda.synchronize()` to capture `prefill_ms`, `total_decode_ms`, and `decode_ms_per_token` per `RunResult`. These feed the memory-vs-latency Pareto figure in `experiments/plot.py::plot_latency_pareto`.

## Experiment pipeline

`experiments/runner.py` drives the strategy × budget × prompt sweep:

1. For each prompt, run a reference generation with the `standard` cache to collect ground-truth logits.
2. For each (strategy, budget) combination, run the prompt's reference cache (memory metrics) plus a teacher-forced replay against the ref logits (KL divergence + perplexity + latency timings).
3. Each (strategy, budget, prompt) tuple becomes one `RunResult`.

`experiments/plot.py` consumes `RunResult` lists and produces 6 study figures:
- `pareto.png` — memory vs perplexity, with bootstrap 95% CI bands
- `kl_pareto.png` — memory vs KL divergence, with CI bands
- `latency_pareto.png` — memory vs decode-ms-per-token
- `survival_heatmap.png` — per-layer token survival rate per strategy
- `layer_type_breakdown.png` — sliding-vs-global memory split
- `category_breakdown.png` — per-prompt-category perplexity with CI error bars

Plus diagnostic figures: `perplexity_delta.png`, `kl_delta.png`, `eviction_pressure.png`, `prompt_length_effect.png`.

Bootstrap CI helper (`_bootstrap_ci`) uses a seeded RNG (`np.random.default_rng(42)`) so figures are byte-identical across runs.

## Weight loading

Two loaders, both in `weights.py`:

**`load_hf_weights`** — Gemma 4 E4B. Downloads from HuggingFace Hub via `snapshot_download`, loads safetensors, maps HF parameter names to KVLens names. Key mappings:
- PLE embedding: single `[262144, 10752]` tensor reshaped to `[262144, 42, 256]`
- Q/K/V norms: per-layer RMSNorm weights
- Layer scalar: per-layer learnable multiplier
- Tied embeddings: LM head shares weights with token embedding
- KV-sharing layers: no K/V projections in the checkpoint (expected and handled)

**`load_gemma2_weights`** — Gemma 2 2B. Simpler HF layout with no PLE, no KV sharing. Direct `model.*` prefix stripping. Raises on any missing key.

Both loaders call `verify_state_dict_shapes` before `load_state_dict`, ensuring every parameter matches exactly before any weights are applied.

**RMSNorm convention:** All Gemma variants store norm weights as deltas from zero. Effective scale = `1 + stored_weight`, not `stored_weight` directly. This applies to every RMSNorm in the model.
