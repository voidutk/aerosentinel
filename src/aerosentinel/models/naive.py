"""Naive baselines: a constant predictor and ridge regression on the flattened window."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from aerosentinel.data.prepare import WindowData
from aerosentinel.metrics import rmse

F32 = NDArray[np.float32]
F64 = NDArray[np.float64]


class ConstantRUL:
    """Predicts the mean capped RUL label of the train-split training windows."""

    name = "constant"

    def __init__(self) -> None:
        self.value: float | None = None
        self.channels: list[str] = []

    def fit(self, train: WindowData, val: WindowData, seed: int) -> dict[str, Any]:
        """Set the constant to the train-split window-label mean (val is unused)."""
        self.channels = list(train.channels)
        self.value = float(np.mean(train.window_labels(), dtype=np.float64))
        return {"value": self.value}

    def predict(self, x: F32, regime_last: NDArray[np.int64]) -> F32:
        """Return the constant for every window."""
        if self.value is None:
            raise RuntimeError("model is not fitted")
        return np.full(x.shape[0], self.value, dtype=np.float32)

    def input_names(self) -> list[str]:
        """No inputs are used."""
        return []

    def n_params(self) -> int:
        """One parameter."""
        return 1

    def save(self, run_dir: Path) -> None:
        """Save the constant."""
        (run_dir / "constant.json").write_text(json.dumps({"value": self.value}))

    def load(self, run_dir: Path) -> None:
        """Load the constant."""
        self.value = float(json.loads((run_dir / "constant.json").read_text())["value"])


def _flat(x: F32) -> F64:
    return x.reshape(x.shape[0], -1).astype(np.float64)


class RidgeLastWindow:
    """Ridge regression on the flattened T x C window, alpha chosen on validation units.

    The fit streams over window batches and accumulates the normal equations, so no
    flattened window matrix is ever built (SPEC §12). It is equivalent to
    ``sklearn.linear_model.Ridge(alpha, fit_intercept=True)``.
    """

    name = "ridge"

    def __init__(self, alphas: Sequence[float] = (0.1, 1.0, 10.0, 100.0)) -> None:
        self.alphas = [float(a) for a in alphas]
        self.coef: F64 | None = None
        self.intercept: float = 0.0
        self.alpha: float | None = None
        self.channels: list[str] = []
        self.window: int = 0

    @staticmethod
    def solve(xtx: F64, xty: F64, sx: F64, sy: float, n: int, alpha: float) -> tuple[F64, float]:
        """Solve centered ridge from accumulated sums.

        Parameters
        ----------
        xtx, xty : numpy.ndarray
            ``X^T X`` [D, D] and ``X^T y`` [D] of the raw (uncentered) features.
        sx, sy : numpy.ndarray, float
            Column sums of X and the sum of y.
        n : int
            Number of rows.
        alpha : float
            L2 penalty (not applied to the intercept).

        Returns
        -------
        (coef, intercept)
        """
        mx, my = sx / n, sy / n
        cxx = xtx - n * np.outer(mx, mx)
        cxy = xty - n * mx * my
        coef: F64 = np.asarray(np.linalg.solve(cxx + alpha * np.eye(len(mx)), cxy), np.float64)
        return coef, float(my - mx @ coef)

    def fit(self, train: WindowData, val: WindowData, seed: int) -> dict[str, Any]:
        """Accumulate normal equations on train windows; pick alpha by val-unit RMSE."""
        self.channels = list(train.channels)
        self.window = train.index.window
        d = self.window * len(self.channels)
        xtx, xty, sx = np.zeros((d, d)), np.zeros(d), np.zeros(d)
        sy, n = 0.0, 0
        for x, y, _ in train.batches():
            xf, yf = _flat(x), y.astype(np.float64)
            xtx += xf.T @ xf
            xty += xf.T @ yf
            sx += xf.sum(axis=0)
            sy += float(yf.sum())
            n += len(yf)
        yva = val.window_labels()
        val_rmse: dict[str, float] = {}
        best: tuple[float, float, F64, float] | None = None
        for a in self.alphas:
            coef, b = self.solve(xtx, xty, sx, sy, n, a)
            pred = np.concatenate([_flat(x) @ coef + b for x, _, _ in val.batches()])
            score = rmse(yva, pred)
            val_rmse[str(a)] = score
            if best is None or score < best[0]:
                best = (score, a, coef, b)
        assert best is not None
        _, self.alpha, self.coef, self.intercept = best
        return {"alpha": self.alpha, "val_rmse_by_alpha": val_rmse}

    def predict(self, x: F32, regime_last: NDArray[np.int64]) -> F32:
        """Predict RUL for windows [B, T, C]."""
        if self.coef is None:
            raise RuntimeError("model is not fitted")
        return (_flat(x) @ self.coef + self.intercept).astype(np.float32)

    def input_names(self) -> list[str]:
        """Flattened inputs ``<sensor>__t<k>``."""
        return [f"{c}__t{k}" for k in range(self.window) for c in self.channels]

    def n_params(self) -> int:
        """Coefficients plus intercept."""
        return 0 if self.coef is None else int(self.coef.size + 1)

    def save(self, run_dir: Path) -> None:
        """Save coefficients and metadata."""
        if self.coef is None:
            raise RuntimeError("model is not fitted")
        np.save(run_dir / "ridge_coef.npy", self.coef)
        meta = {
            "intercept": self.intercept,
            "alpha": self.alpha,
            "channels": self.channels,
            "window": self.window,
        }
        (run_dir / "ridge.json").write_text(json.dumps(meta))

    def load(self, run_dir: Path) -> None:
        """Load a model saved by :meth:`save`."""
        self.coef = np.load(run_dir / "ridge_coef.npy")
        meta = json.loads((run_dir / "ridge.json").read_text())
        self.intercept = float(meta["intercept"])
        self.alpha = meta["alpha"]
        self.channels = list(meta["channels"])
        self.window = int(meta["window"])
