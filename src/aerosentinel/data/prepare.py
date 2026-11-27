"""C-MAPSS preprocessing pipeline shared by training and evaluation.

``prepare_dev`` reads only the train file: it splits units into train/val, fits the
regime normalizer and the sensor selector on **train-split units only**, and returns
per-unit arrays plus window indices. ``prepare_test`` applies the fitted preprocessor to
the official test file; only :mod:`aerosentinel.evaluate` calls it.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from aerosentinel.data.cmapss import (
    RUL,
    UNIT,
    RegimeNormalizer,
    SensorSelector,
    add_rul,
    load_test,
    load_train,
    unit_arrays,
)
from aerosentinel.data.splits import train_val_units
from aerosentinel.data.windows import WindowIndex, build_window_index, last_window

F32 = NDArray[np.float32]


@dataclass
class Preprocessor:
    """Fitted regime normalizer + sensor selector (fit on training-split units only)."""

    normalizer: RegimeNormalizer
    selector: SensorSelector
    fit_units: list[int] = field(default_factory=list)

    @property
    def channels(self) -> list[str]:
        """Model input channels: the kept regime-normalized sensors."""
        return list(self.selector.kept_)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Regime-normalize ``df`` (adds a ``regime`` column)."""
        return self.normalizer.transform(df)


def fit_preprocessor(
    train_df: pd.DataFrame, n_regimes: int, var_threshold: float, seed: int = 0
) -> Preprocessor:
    """Fit regime normalization, THEN the variance threshold, on ``train_df`` only.

    Parameters
    ----------
    train_df : pandas.DataFrame
        Rows of training-split units only.
    n_regimes : int
        KMeans k (6 for FD002/FD004, 1 otherwise).
    var_threshold : float
        Variance threshold applied after regime normalization.
    seed : int, default 0
        KMeans random state.

    Returns
    -------
    Preprocessor
        Fitted preprocessor.
    """
    norm = RegimeNormalizer(n_regimes=n_regimes, seed=seed).fit(train_df)
    sel = SensorSelector(threshold=var_threshold).fit(norm.transform(train_df))
    return Preprocessor(norm, sel, sorted(int(u) for u in train_df[UNIT].unique()))


@dataclass
class WindowData:
    """Per-unit float32 arrays, labels, regimes, and a window index over them."""

    units: list[int]
    arrays: list[F32]
    labels: list[F32]
    regimes: list[NDArray[np.int64]]
    index: WindowIndex
    channels: list[str]

    def window_labels(self) -> F32:
        """Label at the end of every indexed window."""
        return np.array(
            [self.labels[s][e] for s, e in zip(self.index.segment, self.index.end, strict=True)],
            dtype=np.float32,
        )

    def batches(self, batch_size: int = 4096) -> Iterator[tuple[F32, F32, NDArray[np.int64]]]:
        """Yield ``(x [B,T,C], y [B], regime_at_window_end [B])`` gathered batch by batch.

        Only one batch of windows exists in memory at a time.
        """
        idx = self.index
        t = idx.window
        for b0 in range(0, len(idx), batch_size):
            seg = idx.segment[b0 : b0 + batch_size]
            st = idx.start[b0 : b0 + batch_size]
            x = np.stack([self.arrays[s][a : a + t] for s, a in zip(seg, st, strict=True)])
            y = np.array([self.labels[s][a + t - 1] for s, a in zip(seg, st, strict=True)])
            r = np.array([self.regimes[s][a + t - 1] for s, a in zip(seg, st, strict=True)])
            yield x.astype(np.float32, copy=False), y.astype(np.float32), r.astype(np.int64)


def _window_data(
    df: pd.DataFrame, pre: Preprocessor, window: int, stride: int, label: str | None
) -> WindowData:
    units, xs, ys, rs = unit_arrays(pre.transform(df), pre.channels, label)
    arrays = [x.astype(np.float32) for x in xs]
    labels = [y.astype(np.float32) for y in ys]
    index = build_window_index([len(a) for a in arrays], window, stride)
    return WindowData(units, arrays, labels, rs, index, pre.channels)


@dataclass
class DevData:
    """Everything training needs: the fitted preprocessor and train/val window data."""

    pre: Preprocessor
    train: WindowData
    val: WindowData


def prepare_dev(
    raw_dir: Path,
    subset: str,
    *,
    cap: int | None,
    window: int,
    stride: int,
    val_frac: float,
    split_seed: int,
    n_regimes: int,
    var_threshold: float,
) -> DevData:
    """Load the train file, split by unit, fit preprocessing on train-split units only.

    Returns
    -------
    DevData
        Preprocessor plus train and val :class:`WindowData` (labels capped at ``cap``).
    """
    df = add_rul(load_train(raw_dir, subset), cap=cap)
    tr_units, va_units = train_val_units(df[UNIT].unique().tolist(), val_frac, split_seed)
    tr_df = df[df[UNIT].isin(tr_units)]
    va_df = df[df[UNIT].isin(va_units)]
    pre = fit_preprocessor(tr_df, n_regimes, var_threshold, seed=split_seed)
    return DevData(
        pre,
        _window_data(tr_df, pre, window, stride, RUL),
        _window_data(va_df, pre, window, stride, RUL),
    )


@dataclass
class TestData:
    """Last (edge-padded if short) window of every official test unit."""

    units: list[int]
    x: F32
    regime_last: NDArray[np.int64]
    n_padded: NDArray[np.int64]
    lengths: NDArray[np.int64]
    rul_true: NDArray[np.int64]


def prepare_test(raw_dir: Path, subset: str, pre: Preprocessor, window: int) -> TestData:
    """Apply the fitted preprocessor to the official test file; keep each unit's last window.

    Only :mod:`aerosentinel.evaluate` calls this.
    """
    df, rul = load_test(raw_dir, subset)
    units, xs, _, rs = unit_arrays(pre.transform(df), pre.channels, label=None)
    wins, pads = zip(*(last_window(x, window) for x in xs), strict=True)
    return TestData(
        units=units,
        x=np.stack(wins).astype(np.float32),
        regime_last=np.array([r[-1] for r in rs], dtype=np.int64),
        n_padded=np.array(pads, dtype=np.int64),
        lengths=np.array([len(x) for x in xs], dtype=np.int64),
        rul_true=rul,
    )
