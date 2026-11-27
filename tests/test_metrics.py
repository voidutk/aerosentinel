"""Metrics against hand-computed values (SPEC §2, §11)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from aerosentinel.metrics import mae, phm21_score, point_metrics, rmse, saxena_score


def test_saxena_hand_examples() -> None:
    e1 = math.e - 1
    assert saxena_score([50], [60]) == pytest.approx(e1)  # d = +10 (late) -> e^1 - 1
    assert saxena_score([50], [37]) == pytest.approx(e1)  # d = -13 (early) -> e^1 - 1
    assert saxena_score([50], [50]) == 0.0
    # Late is punished harder than early for the same |d|.
    assert saxena_score([50], [60]) > saxena_score([50], [40])


def test_saxena_sum_vs_mean() -> None:
    y, p = [50, 50, 50], [60, 37, 50]
    e1 = math.e - 1
    assert saxena_score(y, p, "sum") == pytest.approx(2 * e1)
    assert saxena_score(y, p, "mean") == pytest.approx(2 * e1 / 3)
    with pytest.raises(ValueError):
        saxena_score(y, p, "median")  # type: ignore[arg-type]


def test_rmse_mae_phm21() -> None:
    y, p = np.array([10.0, 20.0, 30.0, 40.0]), np.array([12.0, 18.0, 33.0, 40.0])
    # errors: +2, -2, +3, 0
    assert rmse(y, p) == pytest.approx(math.sqrt((4 + 4 + 9 + 0) / 4))
    assert mae(y, p) == pytest.approx(7 / 4)
    s = (math.expm1(0.2) + math.expm1(2 / 13) + math.expm1(0.3) + 0) / 4
    assert phm21_score(y, p) == pytest.approx(0.5 * rmse(y, p) + 0.5 * s)
    m = point_metrics(y, p)
    assert set(m) == {"rmse", "mae", "score_sum", "score_mean", "phm21"}
    assert m["score_sum"] == pytest.approx(4 * s)


def test_shape_checks() -> None:
    with pytest.raises(ValueError):
        rmse([1, 2], [1])
    with pytest.raises(ValueError):
        mae([], [])
