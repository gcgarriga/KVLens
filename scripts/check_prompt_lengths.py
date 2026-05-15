"""Print word-count and category distribution histograms for the prompt files.

Useful as a quick sanity check before kicking off a study run. The script does
not require the Gemma tokenizer — it uses word count as a proxy (1 token ≈ 0.75
words for these prompts).

Run:
    python scripts/check_prompt_lengths.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROMPT_FILES = (
    "eval_prompts.txt",
    "length_ablation_short.txt",
    "length_ablation_long.txt",
    "length_ablation_8k.txt",
)


def _word_count(text: str) -> int:
    return len(text.split())


def _bucket(counts: list[int], bins: tuple[int, ...]) -> dict[str, int]:
    bucket: dict[str, int] = {}
    for c in counts:
        for i, edge in enumerate(bins):
            if c < edge:
                key = "0" if i == 0 else f"{bins[i - 1]}"
                bucket[f"{key}-{edge}"] = bucket.get(f"{key}-{edge}", 0) + 1
                break
        else:
            bucket[f">={bins[-1]}"] = bucket.get(f">={bins[-1]}", 0) + 1
    return bucket


def report(path: Path) -> None:
    if not path.exists():
        print(f"{path.name:30s}  (missing)")
        return
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    rows = []
    for line in lines:
        cat, _, prompt = line.partition(":")
        rows.append((cat.strip(), _word_count(prompt.strip())))
    if not rows:
        print(f"{path.name:30s}  (empty)")
        return

    counts = [c for _, c in rows]
    cats: dict[str, int] = {}
    for cat, _ in rows:
        cats[cat] = cats.get(cat, 0) + 1

    bins = (50, 200, 500, 1000, 2000, 4000)
    distribution = _bucket(counts, bins)

    median = sorted(counts)[len(counts) // 2]
    print(
        f"{path.name:30s}  n={len(rows):3d}  "
        f"words [{min(counts):5d}-{max(counts):5d}]  median_words={median:5d}"
    )
    print(f"  categories: {cats}")
    print(f"  word buckets: {distribution}")


def _prompt_paths(argv: list[str], prompts_dir: Path) -> list[Path]:
    if argv:
        return [Path(arg) for arg in argv]
    return [prompts_dir / name for name in PROMPT_FILES]


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    repo_root = Path(__file__).resolve().parent.parent
    prompts_dir = repo_root / "prompts"
    paths = _prompt_paths(args, prompts_dir)
    parents = {path.parent for path in paths}
    target = parents.pop() if len(parents) == 1 else "explicit paths"
    print(f"Inspecting prompt files in {target}")
    print("-" * 60)
    for path in paths:
        report(path)


if __name__ == "__main__":
    main()
