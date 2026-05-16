#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

OUT=results/gemma4_length_ablation
mkdir -p "${OUT}"

STRATS=(
  standard h2o snapkv streaming pyramidkv
  hybrid_h2o hybrid_snapkv hybrid_streaming
)
STRATS_CSV=$(IFS=,; echo "${STRATS[*]}")

run_one() {
  local prompts=$1
  local label=$2
  local results_path="${OUT}/${label}.json"
  local log_path="${OUT}/${label}.log"

  {
    echo "Gemma 4 prompt-length ablation"
    echo "  label=${label}"
    echo "  prompts=${prompts}"
    echo "  strategies=${STRATS_CSV}"
    echo "  memory_budget_mb=512"
    echo "  max_tokens=32"
    echo

    kvlens experiment \
      --prompts "${prompts}" \
      --repo-id google/gemma-4-e4b \
      --model-type gemma4 \
      --strategies "${STRATS_CSV}" \
      --memory-budget 512 \
      --max-tokens 32 \
      --output "${results_path}"
  } 2>&1 | tee "${log_path}"
}

head -5 prompts/eval_prompts.txt > /tmp/gemma4_length_1k_prompts.txt

run_one prompts/length_ablation_short.txt length_256
run_one /tmp/gemma4_length_1k_prompts.txt length_1k
run_one prompts/length_ablation_long.txt length_2k
