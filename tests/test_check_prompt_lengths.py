from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_check_prompt_lengths():
    src = Path(__file__).resolve().parent.parent / "scripts" / "check_prompt_lengths.py"
    spec = importlib.util.spec_from_file_location("check_prompt_lengths", src)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_main_reports_explicit_prompt_file(tmp_path: Path, capsys) -> None:
    module = _load_check_prompt_lengths()
    prompt_file = tmp_path / "phase_e_prompts.txt"
    prompt_file.write_text("technical: one two three\n")

    module.main([str(prompt_file)])

    output = capsys.readouterr().out
    assert str(tmp_path) in output
    assert "phase_e_prompts.txt" in output
    assert "n=  1" in output
    assert "median_words=    3" in output
