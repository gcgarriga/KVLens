"""Regression checks for project dependency metadata."""

from __future__ import annotations

import tomllib
from pathlib import Path


def test_dev_dependencies_keep_numpy_compatible_with_python_311() -> None:
    pyproject = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())

    assert "numpy>=1.26,<2.5" in pyproject["project"]["optional-dependencies"]["dev"]
