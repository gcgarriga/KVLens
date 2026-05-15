from pathlib import Path

CATEGORIES_STUDY = {"retrieval", "long_context", "reasoning", "technical", "creative"}
EXPECTED_PROMPT_FILES = {
    "length_ablation_8k.txt",
    "length_ablation_long.txt",
    "length_ablation_short.txt",
    "eval_prompts.txt",
}


def _rows(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _prompt_word_counts(path: Path) -> list[int]:
    counts = []
    for row in _rows(path):
        _, _, prompt = row.partition(":")
        counts.append(len(prompt.split()))
    return counts


def _categories_in(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in _rows(path):
        cat, _, _ = row.partition(":")
        counts[cat] = counts.get(cat, 0) + 1
    return counts


def test_prompt_manifest_contains_only_publication_inputs() -> None:
    assert {path.name for path in Path("prompts").glob("*.txt")} == EXPECTED_PROMPT_FILES


def test_length_ablation_prompts_have_expected_categories() -> None:
    short_cats = _categories_in(Path("prompts/length_ablation_short.txt"))
    assert short_cats == {"retrieval": 5}

    for name in ("length_ablation_long.txt", "length_ablation_8k.txt"):
        cats = _categories_in(Path("prompts") / name)
        assert set(cats.keys()) == CATEGORIES_STUDY
        assert all(count == 1 for count in cats.values())


def test_eval_prompts_has_50_balanced_prompts() -> None:
    cats = _categories_in(Path("prompts/eval_prompts.txt"))
    assert set(cats.keys()) == CATEGORIES_STUDY
    assert sum(cats.values()) == 50
    assert all(count == 10 for count in cats.values())


def test_eval_prompts_has_mixed_lengths() -> None:
    counts = _prompt_word_counts(Path("prompts/eval_prompts.txt"))
    short = sum(1 for c in counts if c < 200)
    medium = sum(1 for c in counts if 700 <= c <= 1700)
    long = sum(1 for c in counts if c >= 2000)
    assert short == 0, "primary eval set should exclude short smoke/stress prompts"
    assert medium >= 25, "primary eval set should include the 1k tier"
    assert long >= 25, "primary eval set should include the 4k tier"


def test_length_ablation_8k_has_five_balanced_long_prompts() -> None:
    path = Path("prompts/length_ablation_8k.txt")
    counts = _prompt_word_counts(path)
    cats = _categories_in(path)

    assert len(counts) == 5
    assert set(cats.keys()) == CATEGORIES_STUDY
    assert min(counts) >= 6000
    assert max(counts) <= 7000
