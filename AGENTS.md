# AGENTS.md

Public KVLens is a from-scratch PyTorch research engine for studying KV-cache
eviction on Gemma 4 E4B, with a Gemma 2 comparison path. Keep model execution,
attention, cache updates, and measurements explicit and inspectable.

## Commands

| Task | Command |
|---|---|
| Install | `pip install -e ".[dev]"` |
| Test | `pytest -q --tb=short -m "not integration"` |
| Lint | `ruff check .` |
| Format check | `ruff format --check .` |
| Type check | `mypy src/kvlens` |
| Compile check | `python -m compileall -q scripts src/kvlens tests` |
| Gemma 4 sweep | `bash scripts/run_gemma4_sweep.sh` |
| Gemma 4 hybrid canary | `bash scripts/run_gemma4_hybrid_canary.sh` |
| Gemma 4 length ablation | `bash scripts/run_gemma4_length_ablation.sh` |
| Gemma 2 comparison | `bash scripts/run_gemma2_comparison.sh` |
| Validate results | `python scripts/validate_results.py --profile gemma4 --results results/gemma4_full_sweep/results.json --figures results/gemma4_full_sweep/figures` |
| Generate figures | `python scripts/generate_study_figures.py` |

Use `prompts/eval_prompts.txt` for public prompts. Public result paths are
`results/gemma4_full_sweep`, `results/gemma4_hybrid_canary`,
`results/gemma4_length_ablation`, and `results/gemma2_comparison`. Set
`HF_TOKEN` from `.env.example`, which contains `HF_TOKEN=`, only when gated
Gemma weights are needed.

## Rules

- Keep the implementation pure PyTorch: no HuggingFace `transformers`, vLLM,
  `nn.MultiheadAttention`, or abstractions that hide attention/cache mechanics.
- Keep Q/K/V projection, RoPE, masking, softmax, cache reads, and cache updates
  explicit in project code.
- Preserve module boundaries across config, model, attention, cache strategies,
  generation, experiments, plotting, instrumentation, tokenizer, and weights.
- Use `Gemma4Config.layer_params(layer_idx)` for layer behavior instead of
  branching directly on layer type in attention.
- Keep cache strategies swappable through `create_cache()` and the common
  `update/get/reset/seq_length/key_offset/memory_bytes` interface.
- Use typed public signatures and `torch.Tensor` for tensor arguments.
- Preserve cheap disabled instrumentation paths with an early enabled check.
- Pass `kvlens experiment --strategies` as one comma-separated string.

## Boundaries

- Always: run relevant tests, lint, format-check, and type-check before
  publishing changes.
- Ask first: new dependencies, weight-loading changes, model math changes, cache
  interface changes, or committed result artifact updates.
- Never: commit model weights, local tokens, `.env` files, credentials, or
  provider-specific setup.
- Never: replace explicit attention/cache code with inference-framework helpers.

## Canonical examples

- Attention: `src/kvlens/attention.py`
- Cache factory: `src/kvlens/cache/__init__.py`
- Generation: `src/kvlens/generation.py`
- Experiments: `src/kvlens/experiments/runner.py`
- Plotting: `src/kvlens/experiments/plot.py`
- CPU fixtures: `tests/conftest.py`
