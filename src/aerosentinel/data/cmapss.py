"""C-MAPSS loading, RUL labelling, regime normalization and sensor selection (SPEC §1.1, §1.3).

Order of preprocessing (fixed by SPEC §1.3 item 5):

1. Regime normalization: KMeans (k=6) on the 3 operating settings for FD002/FD004, then a
   per-regime z-score of every sensor. FD001/FD003 use one global regime.
2. THEN the near-constant-sensor variance threshold, on the normalized training data.

Both steps are fit on training-split units only. Model channels are the kept normalized
sensors; settings, unit, cycle and RUL never become channels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.cluster import KMeans

UNIT, CYCLE, RUL = "unit", "cycle", "RUL"
SETTINGS: tuple[str, ...] = ("setting1", "setting2", "setting3")
SENSORS: tuple[str, ...] = tuple(f"s{i}" for i in range(1, 22))
COLUMNS: tuple[str, ...] = (UNIT, CYCLE, *SETTINGS, *SENSORS)
N_COLUMNS = 26
assert len(COLUMNS) == N_COLUMNS

FloatArray = NDArray[np.float64]


def read_cmapss_table(path: Path) -> pd.DataFrame:
    """Read one ``train_FD00x.txt`` / ``test_FD00x.txt`` file.

    The files are whitespace-separated with trailing spaces, so they are parsed with
    ``sep=r"\\s+"`` and must yield exactly 26 columns.

    Parameters
    ----------
    path : pathlib.Path
        Text file.

    Returns
    -------
    pandas.DataFrame
        Columns :data:`COLUMNS`; ``unit``/``cycle`` as int64, the rest float64.

    Raises
    ------
    ValueError
        If the file does not parse into exactly 26 columns.
    """
    df = pd.read_csv(path, sep=r"\s+", header=None)
    if df.shape[1] != N_COLUMNS:
        raise ValueError(f"{path}: expected {N_COLUMNS} columns, got {df.shape[1]}")
    df.columns = list(COLUMNS)
    df = df.astype({c: "float64" for c in (*SETTINGS, *SENSORS)})
    return df.astype({UNIT: "int64", CYCLE: "int64"})


def read_rul_file(path: Path) -> NDArray[np.int64]:
    """Read ``RUL_FD00x.txt``: one true RUL per test unit, in unit order.

    Parameters
    ----------
    path : pathlib.Path
        Text file.

    Returns
    -------
    numpy.ndarray
        int64 array of length n_test_units.

    Raises
    ------
    ValueError
        If the file does not parse into exactly one column.
    """
    df = pd.read_csv(path, sep=r"\s+", header=None)
    if df.shape[1] != 1:
        raise ValueError(f"{path}: expected 1 column, got {df.shape[1]}")
    return df.iloc[:, 0].to_numpy(dtype=np.int64)


def load_train(raw_dir: Path, subset: str) -> pd.DataFrame:
    """Load ``train_<subset>.txt`` (run-to-failure trajectories)."""
    return read_cmapss_table(raw_dir / f"train_{subset}.txt")


def load_test(raw_dir: Path, subset: str) -> tuple[pd.DataFrame, NDArray[np.int64]]:
    """Load ``test_<subset>.txt`` and ``RUL_<subset>.txt``.

    Only :mod:`aerosentinel.evaluate` calls this (official test units are touched only by
    the final evaluation).

    Returns
    -------
    (pandas.DataFrame, numpy.ndarray)
        Truncated test trajectories and the true RUL at each unit's last cycle,
        ordered by sorted unit id.
    """
    df = read_cmapss_table(raw_dir / f"test_{subset}.txt")
    rul = read_rul_file(raw_dir / f"RUL_{subset}.txt")
    n_units = df[UNIT].nunique()
    if len(rul) != n_units:
        raise ValueError(f"{subset}: {len(rul)} RUL values for {n_units} test units")
    return df, rul


def add_rul(df: pd.DataFrame, cap: int | None = 125) -> pd.DataFrame:
    """Add the piecewise-linear RUL label to run-to-failure trajectories.

    ``RUL = max_cycle(unit) - cycle``, then clipped at ``cap``.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain ``unit`` and ``cycle``.
    cap : int or None, default 125
        Upper cap; None means uncapped.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with a float64 ``RUL`` column.
    """
    out = df.copy()
    rul = out.groupby(UNIT)[CYCLE].transform("max") - out[CYCLE]
    if cap is not None:
        rul = rul.clip(upper=cap)
    out[RUL] = rul.astype("float64")
    return out


def cap_truth(rul: NDArray[np.int64] | FloatArray, cap: int | None) -> FloatArray:
    """Return test truth as float, clipped at ``cap`` (None = uncapped)."""
    arr = np.asarray(rul, dtype=np.float64)
    return arr if cap is None else np.minimum(arr, float(cap))


@dataclass
class RegimeNormalizer:
    """Per-operating-regime z-score of the sensors.

    Parameters
    ----------
    n_regimes : int
        1 for FD001/FD003 (global z-score); 6 for FD002/FD004 (KMeans on settings).
    seed : int
        KMeans ``random_state``.
    std_tol : float
        A sensor whose within-regime std is below ``std_tol * (1 + |mean|)`` is treated as
        constant in that regime and maps to exactly 0.
    """

    n_regimes: int = 1
    seed: int = 0
    std_tol: float = 1e-8
    settings_mean_: FloatArray = field(default_factory=lambda: np.zeros(0), repr=False)
    settings_std_: FloatArray = field(default_factory=lambda: np.zeros(0), repr=False)
    kmeans_: KMeans | None = field(default=None, repr=False)
    mean_: FloatArray = field(default_factory=lambda: np.zeros(0), repr=False)
    std_: FloatArray = field(default_factory=lambda: np.zeros(0), repr=False)

    def _settings_z(self, df: pd.DataFrame) -> FloatArray:
        s = df[list(SETTINGS)].to_numpy(dtype=np.float64)
        return (s - self.settings_mean_) / self.settings_std_

    def fit(self, train_df: pd.DataFrame) -> RegimeNormalizer:
        """Fit regimes and per-regime sensor statistics on training-split rows only."""
        if self.n_regimes > 1:
            s = train_df[list(SETTINGS)].to_numpy(dtype=np.float64)
            self.settings_mean_ = s.mean(axis=0)
            sd = s.std(axis=0)
            self.settings_std_ = np.where(sd > 0, sd, 1.0)
            self.kmeans_ = KMeans(n_clusters=self.n_regimes, n_init=10, random_state=self.seed)
            self.kmeans_.fit(self._settings_z(train_df))
        regimes = self.regimes(train_df)
        x = train_df[list(SENSORS)].to_numpy(dtype=np.float64)
        self.mean_ = np.zeros((self.n_regimes, len(SENSORS)))
        self.std_ = np.ones((self.n_regimes, len(SENSORS)))
        for r in range(self.n_regimes):
            xr = x[regimes == r]
            if len(xr) == 0:
                raise ValueError(f"regime {r} has no training rows")
            mu, sd = xr.mean(axis=0), xr.std(axis=0)
            self.mean_[r] = mu
            # Constant-in-regime sensors: inf std -> normalized value exactly 0.
            self.std_[r] = np.where(sd > self.std_tol * (1.0 + np.abs(mu)), sd, np.inf)
        return self

    def regimes(self, df: pd.DataFrame) -> NDArray[np.int64]:
        """Assign each row to a regime (all zeros when ``n_regimes == 1``)."""
        if self.n_regimes == 1:
            return np.zeros(len(df), dtype=np.int64)
        if self.kmeans_ is None:
            raise RuntimeError("RegimeNormalizer is not fitted")
        return np.asarray(self.kmeans_.predict(self._settings_z(df)), dtype=np.int64)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with sensors z-scored by regime and a ``regime`` column added."""
        if self.mean_.size == 0:
            raise RuntimeError("RegimeNormalizer is not fitted")
        out = df.copy()
        r = self.regimes(df)
        x = df[list(SENSORS)].to_numpy(dtype=np.float64)
        out[list(SENSORS)] = (x - self.mean_[r]) / self.std_[r]
        out["regime"] = r
        return out


@dataclass
class SensorSelector:
    """Variance threshold on regime-normalized sensors, fit on training-split rows.

    After a per-regime z-score every non-constant sensor has variance ~1, so with the
    default threshold this drops sensors that are constant within every regime.

    Parameters
    ----------
    threshold : float
        Sensors with training variance <= threshold are dropped.
    """

    threshold: float = 1e-4
    kept_: list[str] = field(default_factory=list)
    dropped_: list[str] = field(default_factory=list)

    def fit(self, train_norm: pd.DataFrame) -> SensorSelector:
        """Fit on regime-normalized training-split rows."""
        var = train_norm[list(SENSORS)].var(axis=0, ddof=0)
        self.kept_ = [s for s in SENSORS if var[s] > self.threshold]
        self.dropped_ = [s for s in SENSORS if s not in self.kept_]
        if not self.kept_:
            raise ValueError("variance threshold dropped every sensor")
        return self


def unit_arrays(
    df: pd.DataFrame, channels: list[str], label: str | None = RUL
) -> tuple[list[int], list[FloatArray], list[FloatArray], list[NDArray[np.int64]]]:
    """Split a frame into per-unit contiguous arrays (sorted by unit, then cycle).

    Parameters
    ----------
    df : pandas.DataFrame
        Normalized frame with ``unit``, ``cycle``, ``regime`` and the channels.
    channels : list of str
        Feature channels (sensors only).
    label : str or None
        Label column, or None for unlabeled test data (labels become NaN).

    Returns
    -------
    (units, X, y, regimes)
        Unit ids; per-unit float32-safe float64 arrays [N_u, C]; per-unit labels [N_u];
        per-unit regime ids [N_u].
    """
    df = df.sort_values([UNIT, CYCLE])
    units: list[int] = []
    xs: list[FloatArray] = []
    ys: list[FloatArray] = []
    rs: list[NDArray[np.int64]] = []
    for _, g in df.groupby(UNIT, sort=True):
        units.append(int(g[UNIT].iloc[0]))
        xs.append(np.ascontiguousarray(g[channels].to_numpy(dtype=np.float64)))
        ys.append(g[label].to_numpy(dtype=np.float64) if label else np.full(len(g), np.nan))
        rs.append(g["regime"].to_numpy(dtype=np.int64))
    return units, xs, ys, rs
