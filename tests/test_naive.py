"""Naive baselines: constant = train-split label mean; ridge alpha chosen on val units."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from aerosentinel.data.prepare import WindowData, prepare_dev
from aerosentinel.data.windows import build_window_index
from aerosentinel.models.naive import ConstantRUL, RidgeLastWindow


def _data(seed: int, n_units: int = 6, n: int = 40, c: int = 3, t: int = 5) -> WindowData:
    rng = np.random.default_rng(seed)
    arrays = [rng.normal(size=(n, c)).astype(np.float32) for _ in range(n_units)]
    w = rng.normal(size=c)
    labels = [(a @ w * 10 + 50).astype(np.float32) for a in arrays]
    idx = build_window_index([n] * n_units, t, 1)
    regs = [np.zeros(n, np.int64) for _ in range(n_units)]
    return WindowData(list(range(n_units)), arrays, labels, regs, idx, ["s2", "s3", "s4"])


def test_constant_is_train_split_window_label_mean(synthetic_root: Path) -> None:
    dev = prepare_dev(
        synthetic_root / "data" / "raw" / "cmapss",
        "FD001",
        cap=125,
        window=10,
        stride=1,
        val_frac=0.2,
        split_seed=0,
        n_regimes=1,
        var_threshold=1e-4,
    )
    m = ConstantRUL()
    m.fit(dev.train, dev.val, seed=0)
    assert m.value == pytest.approx(float(np.mean(dev.train.window_labels(), dtype=np.float64)))
    # Val labels have no influence.
    for lab in dev.val.labels:
        lab[:] = 1e6
    m2 = ConstantRUL()
    m2.fit(dev.train, dev.val, seed=0)
    assert m2.value == m.value
    x = np.zeros((4, 10, len(dev.pre.channels)), np.float32)
    np.testing.assert_allclose(m.predict(x, np.zeros(4, np.int64)), m.value)


def test_streaming_ridge_matches_sklearn() -> None:
    train, val = _data(0), _data(1)
    m = RidgeLastWindow(alphas=[3.0])
    m.fit(train, val, seed=0)
    x = np.concatenate([b for b, _, _ in train.batches()]).reshape(len(train.index), -1)
    y = train.window_labels()
    ref = Ridge(alpha=3.0).fit(x.astype(np.float64), y.astype(np.float64))
    assert m.coef is not None
    np.testing.assert_allclose(m.coef, ref.coef_, rtol=1e-6, atol=1e-8)
    assert m.intercept == pytest.approx(float(ref.intercept_), rel=1e-6)


def test_ridge_alpha_selected_on_val_units_only() -> None:
    train = _data(0)
    val_a, val_b = _data(1), _data(2)
    # Corrupt val_b's labels so a different alpha may win; train is identical in both fits.
    for lab in val_b.labels:
        lab[:] = 0.0
    a = RidgeLastWindow(alphas=[0.01, 1e6])
    info_a = a.fit(train, val_a, seed=0)
    b = RidgeLastWindow(alphas=[0.01, 1e6])
    info_b = b.fit(train, val_b, seed=0)
    # Val RMSEs are computed on the val units' own labels.
    assert info_a["val_rmse_by_alpha"] != info_b["val_rmse_by_alpha"]
    assert a.alpha == 0.01  # informative val -> weak penalty wins
    best_b = min(info_b["val_rmse_by_alpha"], key=info_b["val_rmse_by_alpha"].get)
    assert b.alpha == float(best_b)
