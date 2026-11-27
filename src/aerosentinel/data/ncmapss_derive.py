"""Derived per-unit arrays for N-CMAPSS: cruise runs and healthy-baseline residuals.

Both derivations use only a unit's own inputs (W, X_s and the cycle number). They never read
Y or hs, so they are computed identically for dev and test units. The prepare script opens
the test split with ``allow_test=True`` for this reason only.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import asdict
from typing import Any

import numpy as np
from numpy.typing import NDArray

from aerosentinel.data.baseline_residuals import (
    BaselineConfig,
    extrapolation_share,
    fit_engine_baseline,
    residuals,
)
from aerosentinel.data.masks import CruiseMaskParams, cruise_run
from aerosentinel.data.ncmapss import UnitStore


def mask_hash(params: CruiseMaskParams) -> str:
    """Short hash of the cruise-mask parameters (file name of the cached runs)."""
    return hashlib.sha256(json.dumps(params.to_dict(), sort_keys=True).encode()).hexdigest()[:12]


def unit_cruise_runs(
    alt: NDArray[Any], flights: NDArray[np.int64], params: CruiseMaskParams
) -> NDArray[np.int64]:
    """Cruise run of every flight as unit-row offsets ``[n_flights, 2]`` (-1, -1 if none)."""
    out = np.full((len(flights), 2), -1, dtype=np.int64)
    for i, (_, s, e) in enumerate(flights):
        run = cruise_run(np.asarray(alt[s:e]), params)
        if run is not None:
            out[i] = (s + run[0], s + run[1])
    return out


def compute_cruise_runs(store: UnitStore, params: CruiseMaskParams) -> dict[int, int]:
    """Write ``cruise_<hash>.npy`` for every unit; return accepted flights per unit."""
    alt_col = store.var_names["W"].index("alt")
    accepted = {}
    for u in store.units:
        ua = store.load(u)
        runs = unit_cruise_runs(ua.W[:, alt_col], ua.flights, params)
        np.save(store.unit_dir(u) / f"cruise_{mask_hash(params)}.npy", runs)
        accepted[u] = int((runs[:, 0] >= 0).sum())
    return accepted


def load_cruise_runs(store: UnitStore, unit: int, params: CruiseMaskParams) -> NDArray[np.int64]:
    """Cached cruise runs of one unit (computed by :func:`compute_cruise_runs`)."""
    path = store.unit_dir(unit) / f"cruise_{mask_hash(params)}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing: run scripts/prepare_ncmapss.py")
    runs: NDArray[np.int64] = np.load(path)
    return runs


def compute_residuals(store: UnitStore, cfg: BaselineConfig) -> dict[int, dict[str, Any]]:
    """Fit each unit's baseline on its own first cycles; write ``residuals.npy``.

    Returns
    -------
    dict
        Per unit: baseline cycles, fit rows, fit MSE (standardized units) and the
        W-extrapolation share of later rows.
    """
    cyc_col = store.a_col("cycle")
    out: dict[int, dict[str, Any]] = {}
    for u in store.units:
        ua = store.load(u)
        cycle = np.asarray(ua.A[:, cyc_col])
        base = fit_engine_baseline(ua.W, ua.X_s, cycle, cfg)
        res = residuals(base, ua.W, ua.X_s)
        np.save(store.unit_dir(u) / "residuals.npy", res)
        with (store.unit_dir(u) / "baseline.pkl").open("wb") as f:
            pickle.dump(base, f)
        later = ~np.isin(cycle, base.cycles)
        out[u] = {
            "baseline_cycles": base.cycles,
            "n_fit_rows": base.n_fit_rows,
            "fit_mse_std_units": base.fit_mse,
            "w_extrapolation_share": extrapolation_share(base, ua.W, cycle),
            "mean_abs_residual_baseline_cycles": float(np.abs(res[~later]).mean()),
            "mean_abs_residual_later_cycles": float(np.abs(res[later]).mean())
            if later.any()
            else None,
        }
    (store.dir / "baseline_config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")
    return out
