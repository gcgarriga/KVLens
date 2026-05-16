#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

CANARY_PROMPTS=/tmp/gemma4_hybrid_canary_prompts.txt
head -5 prompts/eval_prompts.txt > "${CANARY_PROMPTS}"

OUT=results/gemma4_hybrid_canary
mkdir -p "${OUT}"

STRATS=(
  standard
  h2o snapkv streaming pyramidkv
  all_layers_h2o all_layers_snapkv all_layers_streaming all_layers_pyramidkv
  proportional_h2o proportional_snapkv proportional_streaming proportional_pyramidkv
  hybrid_h2o hybrid_snapkv hybrid_streaming
)
STRATS_CSV=$(IFS=,; echo "${STRATS[*]}")

RESULTS_PATH="${OUT}/results.json"
FIGURE_DIR="${OUT}/figures"
LOG_PATH="${OUT}/run.log"

{
  echo "Gemma 4 hybrid canary"
  echo "  prompts=${CANARY_PROMPTS}"
  echo "  strategies=${STRATS_CSV}"
  echo "  memory_budget_mb=4,64,512"
  echo "  max_tokens=32"
  echo

  kvlens experiment \
    --prompts "${CANARY_PROMPTS}" \
    --repo-id google/gemma-4-e4b \
    --model-type gemma4 \
    --strategies "${STRATS_CSV}" \
    --memory-budget 4,64,512 \
    --max-tokens 32 \
    --output "${RESULTS_PATH}"

  kvlens plot \
    --input "${RESULTS_PATH}" \
    --output-dir "${FIGURE_DIR}" \
    --model-type gemma4 \
    --budget 64
} 2>&1 | tee "${LOG_PATH}"
