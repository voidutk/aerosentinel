"""Smoke tests: the package imports and the shared utilities behave."""

from __future__ import annotations

import random

import numpy as np
import torch

import aerosentinel
from aerosentinel.utils import get_paths, seed_everything


def test_package_imports() -> None:
    assert isinstance(aerosentinel.__version__, str)


def test_seed_everything_is_reproducible() -> None:
    def draw() -> tuple[float, np.ndarray, torch.Tensor]:
        return random.random(), np.random.rand(3), torch.rand(3)

    seed_everything(123)
    a = draw()
    seed_everything(123)
    b = draw()
    assert a[0] == b[0]
    np.testing.assert_array_equal(a[1], b[1])
    assert torch.equal(a[2], b[2])


def test_paths_resolve_under_project_root() -> None:
    paths = get_paths()
    assert (paths.root / "pyproject.toml").is_file()
    for p in (paths.data_raw, paths.data_processed, paths.results, paths.checksums):
        assert p.is_relative_to(paths.root)
    assert paths.checksums.is_file()
