#!/usr/bin/env bash
# Gemma 4 E4B full sweep — the study-target run.
#
# 16 strategies × 6 iso-memory budgets × 50 prompts = 4,800 runs.
# Approximate time on an A100 40 GB: ~90 min at the default max_tokens=32.
# Run inside tmux so the SSH session can disconnect:
#
#   tmux new-session "bash scripts/run_gemma4_sweep.sh"
#
# Budget grid spans BOTH memory regimes so iso-actual-memory comparisons
# between hybrid-adapted (`h2o`, `snapkv`, `streaming`, `pyramidkv`),
# canonical (`all_layers_*`), and proportional (`proportional_*`) strategies
# are possible. Adapted strategies keep globals full (~250-400 MB on Gemma 4
# due to head_dim=512), so `all_layers_*`/`proportional_*` need budgets large
# enough to reach that floor — hence 128, 256, and 512 MB points alongside
# the small-budget points. The proportional family gives globals a
# proportionally larger byte share than uniform allocation (iso-token across
# all layers), which is a meaningful difference on Gemma 4 only since
# Gemma 2 has homogeneous head_dim.
#
# Usage:
#   bash scripts/run_gemma4_sweep.sh
#   PROMPTS_PATH=prompts/eval_prompts.txt bash scripts/run_gemma4_sweep.sh
set -euo pipefail

if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

PROMPTS_PATH="${PROMPTS_PATH:-prompts/eval_prompts.txt}"
REPO_ID="${REPO_ID:-google/gemma-4-e4b}"
MODEL_TYPE="${MODEL_TYPE:-gemma4}"
STRATEGIES="${STRATEGIES:-standard,h2o,snapkv,streaming,pyramidkv,all_layers_h2o,all_layers_snapkv,all_layers_streaming,all_layers_pyramidkv,proportional_h2o,proportional_snapkv,proportional_streaming,proportional_pyramidkv,hybrid_h2o,hybrid_snapkv,hybrid_streaming}"
MEMORY_BUDGET="${MEMORY_BUDGET:-4,16,64,128,256,512}"
MAX_TOKENS="${MAX_TOKENS:-32}"
SEED="${SEED:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-results/gemma4_full_sweep}"
PLOT_BUDGET="${PLOT_BUDGET:-64}"

DEVICE_ARGS=()
if [[ -n "${DEVICE:-}" ]]; then
  DEVICE_ARGS=(--device "${DEVICE}")
fi

mkdir -p "${OUTPUT_DIR}"
RESULTS_PATH="${OUTPUT_DIR}/results.json"
FIGURE_DIR="${OUTPUT_DIR}/figures"
LOG_PATH="${OUTPUT_DIR}/run.log"

{
  echo "Gemma 4 full sweep"
  echo "  prompts=${PROMPTS_PATH}"
  echo "  repo=${REPO_ID} model_type=${MODEL_TYPE}"
  echo "  memory_budget_mb=${MEMORY_BUDGET}"
  echo "  max_tokens=${MAX_TOKENS}"
  echo

  kvlens experiment \
    --prompts "${PROMPTS_PATH}" \
    --repo-id "${REPO_ID}" \
    --model-type "${MODEL_TYPE}" \
    --strategies "${STRATEGIES}" \
    --memory-budget "${MEMORY_BUDGET}" \
    --max-tokens "${MAX_TOKENS}" \
    --seed "${SEED}" \
    --output "${RESULTS_PATH}" \
    "${DEVICE_ARGS[@]}"

  kvlens plot \
    --input "${RESULTS_PATH}" \
    --output-dir "${FIGURE_DIR}" \
    --model-type "${MODEL_TYPE}" \
    --budget "${PLOT_BUDGET}"

  python scripts/validate_results.py \
    --profile gemma4 \
    --results "${RESULTS_PATH}" \
    --figures "${FIGURE_DIR}"
} 2>&1 | tee "${LOG_PATH}"
