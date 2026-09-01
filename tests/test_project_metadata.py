"""Regression checks for project dependency metadata."""

from __future__ import annotations

import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version


def test_dev_dependencies_keep_numpy_compatible_with_python_311() -> None:
    with (Path(__file__).parents[1] / "pyproject.toml").open("rb") as pyproject_file:
        pyproject = tomllib.load(pyproject_file)

    requirements = (
        Requirement(value) for value in pyproject["project"]["optional-dependencies"]["dev"]
    )
    numpy_requirements = [
        requirement
        for requirement in requirements
        if canonicalize_name(requirement.name) == "numpy"
    ]

    assert numpy_requirements
    for requirement in numpy_requirements:
        assert any(
            specifier.operator == "<" and Version(specifier.version) == Version("2.5")
            for specifier in requirement.specifier
        )
