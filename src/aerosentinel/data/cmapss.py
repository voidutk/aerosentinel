"""C-MAPSS loading, RUL labelling, regime normalization and sensor selection (SPEC §1.1, §1.3).

Order of preprocessing (fixed by SPEC §1.3 item 5):

1. Regime normalization: KMeans (k=6) on the 3 operating settings for FD002/FD004, then a
   per-regime z-score of every sensor. FD001/FD003 use one global regime.
2. THEN sensor selection on the training data: a sensor is dropped if its normalized
   variance is <= the threshold, or if it takes at most ``max_distinct`` distinct raw values
   within every regime (quantized, near-binary sensors such as FD001 s6; a z-score turns
   one flip between two raw levels into a large spike).

Both steps are fit on training-split units only. Official test files can be read only with
an explicit ``allow_test=True``, which only :mod:`aerosentinel.evaluate` passes. Model
channels are the kept normalized sensors; settings, unit, cycle and RUL never become
channels.
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

#: File-name prefixes of the official test trajectories and their true RUL.
TEST_FILE_PREFIXES: tuple[str, ...] = ("test_", "RUL_")


class TestDataAccessError(PermissionError):
    """Official test files were requested without ``allow_test=True`` (SPEC §1.3 item 6)."""

    __test__ = False  # not a pytest test class


def _guard_test_file(path: Path, allow_test: bool) -> None:
    if path.name.startswith(TEST_FILE_PREFIXES) and not allow_test:
        raise TestDataAccessError(
            f"{path.name} is an official test file; only aerosentinel.evaluate may read it "
            "(pass allow_test=True there and nowhere else)"
        )


def read_cmapss_table(path: Path, *, allow_test: bool = False) -> pd.DataFrame:
    """Read one ``train_FD00x.txt`` / ``test_FD00x.txt`` file.

    The files are whitespace-separated with trailing spaces, so they are parsed with
    ``sep=r"\\s+"`` and must yield exactly 26 columns.

    Parameters
    ----------
    path : pathlib.Path
        Text file.
    allow_test : bool, default False
        Must be True to read an official ``test_*`` file.

    Returns
    -------
    pandas.DataFrame
        Columns :data:`COLUMNS`; ``unit``/``cycle`` as int64, the rest float64.

    Raises
    ------
    ValueError
        If the file does not parse into exactly 26 columns.
    TestDataAccessError
        If ``path`` is an official test file and ``allow_test`` is False.
    """
    _guard_test_file(path, allow_test)
    df = pd.read_csv(path, sep=r"\s+", header=None)
    if df.shape[1] != N_COLUMNS:
        raise ValueError(f"{path}: expected {N_COLUMNS} columns, got {df.shape[1]}")
    df.columns = list(COLUMNS)
    df = df.astype({c: "float64" for c in (*SETTINGS, *SENSORS)})
    return df.astype({UNIT: "int64", CYCLE: "int64"})


def read_rul_file(path: Path, *, allow_test: bool = False) -> NDArray[np.int64]:
    """Read ``RUL_FD00x.txt``: one true RUL per test unit, in unit order.

    Parameters
    ----------
    path : pathlib.Path
        Text file.
    allow_test : bool, default False
        Must be True to read an official ``RUL_*`` file.

    Returns
    -------
    numpy.ndarray
        int64 array of length n_test_units.

    Raises
    ------
    ValueError
        If the file does not parse into exactly one column.
    TestDataAccessError
        If ``path`` is an official RUL file and ``allow_test`` is False.
    """
    _guard_test_file(path, allow_test)
    df = pd.read_csv(path, sep=r"\s+", header=None)
    if df.shape[1] != 1:
        raise ValueError(f"{path}: expected 1 column, got {df.shape[1]}")
    return df.iloc[:, 0].to_numpy(dtype=np.int64)


def load_train(raw_dir: Path, subset: str) -> pd.DataFrame:
    """Load ``train_<subset>.txt`` (run-to-failure trajectories)."""
    return read_cmapss_table(raw_dir / f"train_{subset}.txt")


def load_test(
    raw_dir: Path, subset: str, *, allow_test: bool = False
) -> tuple[pd.DataFrame, NDArray[np.int64]]:
    """Load ``test_<subset>.txt`` and ``RUL_<subset>.txt``.

    Official test units are touched only by the final evaluation, so this raises unless
    ``allow_test=True``; only :mod:`aerosentinel.evaluate` passes it.

    Parameters
    ----------
    raw_dir : pathlib.Path
        ``data/raw/cmapss``.
    subset : str
        ``FD001`` ... ``FD004``.
    allow_test : bool, default False
        Explicit permission to read the official test files.

    Returns
    -------
    (pandas.DataFrame, numpy.ndarray)
        Truncated test trajectories and the true RUL at each unit's last cycle,
        ordered by sorted unit id.

    Raises
    ------
    TestDataAccessError
        If ``allow_test`` is False.
    ValueError
        If test unit ids are not exactly ``1..N`` or the RUL file has a different length.
        The RUL file is matched to units by row order, so both checks are needed.
    """
    if not allow_test:
        raise TestDataAccessError(
            f"official {subset} test units can only be loaded by aerosentinel.evaluate"
        )
    df = read_cmapss_table(raw_dir / f"test_{subset}.txt", allow_test=allow_test)
    rul = read_rul_file(raw_dir / f"RUL_{subset}.txt", allow_test=allow_test)
    units = sorted(int(u) for u in df[UNIT].unique())
    if units != list(range(1, len(units) + 1)):
        raise ValueError(f"{subset}: test unit ids are not exactly 1..{len(units)}")
    if len(rul) != len(units):
        raise ValueError(f"{subset}: {len(rul)} RUL values for {len(units)} test units")
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
    """Sensor selection fit on training-split rows, after regime normalization.

    Two rules, applied together:

    * **variance**: normalized training variance <= ``threshold``. After a per-regime
      z-score every non-constant sensor has variance ~1, so this only drops sensors that
      are constant within every regime.
    * **few distinct values**: at most ``max_distinct`` distinct *raw* values within every
      regime. Such quantized sensors survive the variance rule, but the z-score turns one
      flip between their few levels into a large spike.

    Parameters
    ----------
    threshold : float
        Variance threshold on the normalized sensors.
    max_distinct : int
        A sensor with <= this many distinct raw values in every regime is dropped.
        0 disables the rule.
    """

    threshold: float = 1e-4
    max_distinct: int = 2
    kept_: list[str] = field(default_factory=list)
    dropped_: list[str] = field(default_factory=list)
    dropped_by_rule_: dict[str, list[str]] = field(default_factory=dict)

    def fit(self, train_norm: pd.DataFrame, train_raw: pd.DataFrame) -> SensorSelector:
        """Fit on training-split rows.

        Parameters
        ----------
        train_norm : pandas.DataFrame
            Regime-normalized rows (with the ``regime`` column from
            :meth:`RegimeNormalizer.transform`).
        train_raw : pandas.DataFrame
            The same rows before normalization (same index), for the distinct-value rule.
        """
        if not train_norm.index.equals(train_raw.index):
            raise ValueError("train_norm and train_raw must hold the same rows")
        var = train_norm[list(SENSORS)].var(axis=0, ddof=0)
        low_var = [s for s in SENSORS if not var[s] > self.threshold]
        n_distinct = (
            train_raw[list(SENSORS)].groupby(train_norm["regime"].to_numpy()).nunique().max()
        )
        few = [s for s in SENSORS if s not in low_var and n_distinct[s] <= self.max_distinct]
        self.dropped_by_rule_ = {"variance": low_var, "few_distinct": few}
        self.dropped_ = [s for s in SENSORS if s in low_var or s in few]
        self.kept_ = [s for s in SENSORS if s not in self.dropped_]
        if not self.kept_:
            raise ValueError("sensor selection dropped every sensor")
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
