#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

OUT=results/gemma2_comparison
MAX_TOKENS=32
export MAX_TOKENS
mkdir -p "${OUT}"

{
  echo "Gemma 2 long-prompt comparison"
  echo "  prompts=prompts/length_ablation_8k.txt"
  echo "  repo=google/gemma-2-2b model_type=gemma2"
  echo "  strategies=standard,h2o,hybrid_h2o"
  echo "  memory_budget_mb=256"
  echo "  max_tokens=${MAX_TOKENS}"
  echo

  python scripts/check_prompt_lengths.py prompts/length_ablation_8k.txt
  python - <<'PY'
from pathlib import Path
import os

from kvlens.tokenizer import GemmaTokenizer

MAX_POSITIONS = 8192
MAX_TOKENS = int(os.environ["MAX_TOKENS"])
MIN_MEDIAN_TOKENS = 7000
EXPECTED_CATEGORIES = {"retrieval", "long_context", "reasoning", "technical", "creative"}

tokenizer = GemmaTokenizer.from_pretrained("google/gemma-2-2b")
rows = [line for line in Path("prompts/length_ablation_8k.txt").read_text().splitlines() if line]
categories = [row.partition(":")[0].strip() for row in rows]
if len(rows) != 5:
    raise SystemExit(f"expected 5 prompts, got {len(rows)}")
if set(categories) != EXPECTED_CATEGORIES or len(categories) != len(set(categories)):
    raise SystemExit(f"expected one prompt per category, got {categories}")

lengths = []
for row in rows:
    category, _, prompt = row.partition(":")
    token_count = len(tokenizer.encode(prompt.strip()))
    lengths.append(token_count)
    print(f"{category.strip():12s} tokens={token_count}")

median = sorted(lengths)[len(lengths) // 2]
max_with_generation = max(lengths) + MAX_TOKENS
print(f"median_tokens={median} max_positions_with_generation={max_with_generation}")
if median < MIN_MEDIAN_TOKENS:
    raise SystemExit(f"median token count {median} < {MIN_MEDIAN_TOKENS}")
if max_with_generation > MAX_POSITIONS:
    raise SystemExit(
        f"prompt plus generation positions {max_with_generation} exceed {MAX_POSITIONS}"
    )
PY
  echo

  kvlens experiment \
    --prompts prompts/length_ablation_8k.txt \
    --repo-id google/gemma-2-2b \
    --model-type gemma2 \
    --strategies standard,h2o,hybrid_h2o \
    --memory-budget 256 \
    --max-tokens "${MAX_TOKENS}" \
    --output "${OUT}/results.json"
} 2>&1 | tee "${OUT}/run.log"
