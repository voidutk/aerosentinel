"""Point-prediction metrics (SPEC §2). Convention: d = y_pred - y_true (late = positive)."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike


def _pair(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(y_true, dtype=np.float64).reshape(-1)
    p = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if t.shape != p.shape or t.size == 0:
        raise ValueError(f"shape mismatch or empty: {t.shape} vs {p.shape}")
    return t, p


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root mean squared error in cycles."""
    t, p = _pair(y_true, y_pred)
    return float(np.sqrt(np.mean((p - t) ** 2)))


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute error in cycles."""
    t, p = _pair(y_true, y_pred)
    return float(np.mean(np.abs(p - t)))


def saxena_score(
    y_true: ArrayLike, y_pred: ArrayLike, reduction: Literal["sum", "mean"] = "sum"
) -> float:
    """NASA / Saxena asymmetric score.

    With d = y_pred - y_true: ``exp(-d/13) - 1`` if d < 0 (early), ``exp(d/10) - 1`` if
    d >= 0 (late). Late predictions are penalized harder.

    Parameters
    ----------
    y_true, y_pred : array-like
        Truth and prediction.
    reduction : {"sum", "mean"}, default "sum"
        "sum" for C-MAPSS (literature convention), "mean" for N-CMAPSS (PHM 2021).

    Returns
    -------
    float
        The reduced score (lower is better).
    """
    t, p = _pair(y_true, y_pred)
    d = p - t
    s = np.where(d < 0, np.expm1(-d / 13.0), np.expm1(d / 10.0))
    if reduction == "sum":
        return float(s.sum())
    if reduction == "mean":
        return float(s.mean())
    raise ValueError(f"unknown reduction {reduction!r}")


def phm21_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """PHM 2021 combined score ``0.5 * RMSE + 0.5 * mean Saxena score``."""
    return 0.5 * rmse(y_true, y_pred) + 0.5 * saxena_score(y_true, y_pred, "mean")


def point_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """All point metrics as a dict (keys: rmse, mae, score_sum, score_mean, phm21)."""
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "score_sum": saxena_score(y_true, y_pred, "sum"),
        "score_mean": saxena_score(y_true, y_pred, "mean"),
        "phm21": phm21_score(y_true, y_pred),
    }
