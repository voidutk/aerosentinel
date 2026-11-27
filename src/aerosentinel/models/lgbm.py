"""LightGBM on engineered window features (SPEC §5.1 item 1)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
from numpy.typing import NDArray

from aerosentinel.data.prepare import WindowData

F32 = NDArray[np.float32]
STATS: tuple[str, ...] = ("last", "mean", "std", "slope", "min", "max")


def feature_names(channels: list[str]) -> list[str]:
    """Feature names, ``<sensor>__<stat>`` for each stat and sensor, then ``regime_id``."""
    return [f"{c}__{s}" for s in STATS for c in channels] + ["regime_id"]


def window_features(x: F32, regime_last: NDArray[np.int64]) -> NDArray[np.float32]:
    """Per-window last, mean, std, slope (least squares), min, max of each channel + regime.

    Parameters
    ----------
    x : numpy.ndarray
        Windows [B, T, C].
    regime_last : numpy.ndarray
        Regime id at the window end [B].

    Returns
    -------
    numpy.ndarray
        Features [B, 6C + 1] in :func:`feature_names` order.
    """
    t = np.arange(x.shape[1], dtype=np.float64)
    tc = t - t.mean()
    xd = x.astype(np.float64)
    slope = np.einsum("t,btc->bc", tc, xd - xd.mean(axis=1, keepdims=True)) / (tc @ tc)
    feats = [xd[:, -1], xd.mean(1), xd.std(1), slope, xd.min(1), xd.max(1)]
    return np.concatenate([*feats, regime_last[:, None].astype(np.float64)], axis=1).astype(
        np.float32
    )


def _features(data: WindowData) -> tuple[NDArray[np.float32], F32]:
    fx, fy = [], []
    for x, y, r in data.batches():
        fx.append(window_features(x, r))
        fy.append(y)
    return np.concatenate(fx), np.concatenate(fy)


class LGBMModel:
    """LightGBM regressor with early stopping on validation units."""

    name = "lgbm"

    def __init__(self, params: dict[str, Any], early_stopping_rounds: int = 200) -> None:
        self.params = dict(params)
        self.early_stopping_rounds = early_stopping_rounds
        self.booster: lgb.Booster | None = None
        self.channels: list[str] = []

    def fit(self, train: WindowData, val: WindowData, seed: int) -> dict[str, Any]:
        """Fit on train-split windows; early-stop on val-unit windows."""
        self.channels = list(train.channels)
        xtr, ytr = _features(train)
        xva, yva = _features(val)
        names = feature_names(self.channels)
        params = {
            "objective": "regression",
            "verbosity": -1,
            "seed": seed,
            "deterministic": True,
            "force_row_wise": True,
            "num_threads": int(os.environ.get("OMP_NUM_THREADS", "0")),
            **self.params,
        }
        n_rounds = int(params.pop("n_estimators", 5000))
        dtr = lgb.Dataset(xtr, ytr, feature_name=names, categorical_feature=["regime_id"])
        dva = lgb.Dataset(xva, yva, reference=dtr)
        self.booster = lgb.train(
            params,
            dtr,
            num_boost_round=n_rounds,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)],
        )
        return {"best_iteration": int(self.booster.best_iteration)}

    def _booster(self) -> lgb.Booster:
        if self.booster is None:
            raise RuntimeError("model is not fitted")
        return self.booster

    def predict(self, x: F32, regime_last: NDArray[np.int64]) -> F32:
        """Predict RUL for windows [B, T, C]."""
        b = self._booster()
        pred = b.predict(window_features(x, regime_last), num_iteration=b.best_iteration)
        return np.asarray(pred, dtype=np.float32)

    def input_names(self) -> list[str]:
        """Engineered feature names."""
        return feature_names(self.channels)

    def n_params(self) -> int:
        """Total number of leaves over the trees used at prediction."""
        dump = self._booster().dump_model(num_iteration=self._booster().best_iteration)
        return int(sum(t["num_leaves"] for t in dump["tree_info"]))

    def save(self, run_dir: Path) -> None:
        """Save the booster (text) and the channel list."""
        b = self._booster()
        b.save_model(str(run_dir / "lgbm.txt"), num_iteration=b.best_iteration)
        (run_dir / "channels.json").write_text(json.dumps(self.channels))

    def load(self, run_dir: Path) -> None:
        """Load a booster saved by :meth:`save`."""
        self.booster = lgb.Booster(model_file=str(run_dir / "lgbm.txt"))
        self.channels = json.loads((run_dir / "channels.json").read_text())
