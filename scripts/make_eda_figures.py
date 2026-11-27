"""N-CMAPSS EDA figures (Phase 2) -> ``reports/figures/ncmapss_*.png``.

Usage: ``uv run python scripts/make_eda_figures.py``

1. ``ncmapss_flight_profiles.png``: altitude and Mach of one mid-life flight per flight class
   (DS03 dev units, because every DS02 dev unit is class 3; see the Fc audit).
2. ``ncmapss_sensors_over_life.png``: per-cycle cruise medians of raw sensors for one DS02
   dev unit, with its hs onset.
3. ``ncmapss_hs_onset.png``: healthy vs abnormal cycles per DS02 dev unit.
4. ``ncmapss_residuals_over_life.png``: per-cycle median healthy-baseline residuals of the
   DS02 dev units, aligned at the hs onset. Hypothesis: they drift with degradation.
5. ``ncmapss_altitude_dev_vs_test.png``: altitude distribution of DS02 dev units vs test units
   14 and 15. This opens the test split for W (altitude) only; no labels are read.

Also writes ``reports/results/ncmapss-ds02_data_eda.json`` with the numbers the captions cite.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from aerosentinel.data.masks import DEFAULT_MASK
from aerosentinel.data.ncmapss import (
    UnitStore,
    decode_names,
    flight_descriptors_of_test_units,
)
from aerosentinel.data.ncmapss_derive import load_cruise_runs
from aerosentinel.runs import git_info
from aerosentinel.utils import get_paths
from aerosentinel.viz import NEUTRAL, SERIES, TEXT_2, apply_style, save

EXAMPLE_UNIT = 2
RESIDUAL_SENSORS = ("T48", "T50", "P50", "Wf")  # turbine-section temps/pressure + fuel flow
RAW_SENSORS = ("T48", "T50", "P50", "Nc")


def per_cycle_cruise_median(
    store: UnitStore, unit: int, arr: Any, cols: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Median of ``arr[:, cols]`` over each flight's cruise run (NaN without a run)."""
    ua = store.load(unit)
    runs = load_cruise_runs(store, unit, DEFAULT_MASK)
    cyc = ua.flights[:, 0]
    med = np.full((len(cyc), len(cols)), np.nan)
    for i, (s, e) in enumerate(runs):
        if s >= 0:
            med[i] = np.median(np.asarray(arr[s:e][:, cols]), axis=0)
    return cyc, med


def onset_cycle(store: UnitStore, unit: int) -> int:
    """First cycle with hs == 0 (dev units only)."""
    a = store.load(unit).A
    hs, cyc = a[:, store.a_col("hs")], a[:, store.a_col("cycle")]
    return int(cyc[hs == 0].min())


def fig_flight_profiles(raw: Path, out: Path) -> dict[str, Any]:
    """One mid-life DS03 dev flight per class: altitude and Mach over time."""
    (h5,) = sorted(raw.glob("N-CMAPSS_DS03*.h5"))
    with h5py.File(h5) as f:
        an = decode_names(f["A_var"][()])
        wn = decode_names(f["W_var"][()])
        a = f["A_dev"][:]
        unit, cyc, fc = (a[:, an.index(k)] for k in ("unit", "cycle", "Fc"))
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
        info = {}
        for k, c in enumerate((1, 2, 3)):
            u = int(np.unique(unit[fc == c])[0])
            mid = int(np.median(np.unique(cyc[unit == u])))
            rows = np.flatnonzero((unit == u) & (cyc == mid))
            w = f["W_dev"][rows[0] : rows[-1] + 1]
            t = np.arange(len(w)) / 60.0
            label = f"class {c} (unit {u}, cycle {mid})"
            ax1.plot(t, w[:, wn.index("alt")], color=SERIES[k], lw=1.4, label=label)
            ax2.plot(t, w[:, wn.index("Mach")], color=SERIES[k], lw=1.4)
            info[str(c)] = {"unit": u, "cycle": mid, "minutes": len(w) / 60.0}
    ax1.set_ylabel("altitude [ft]")
    ax2.set_ylabel("Mach [-]")
    ax2.set_xlabel("time in flight [min]")
    ax1.legend(loc="upper right")
    ax1.set_title(
        "Flight profiles by flight class (DS03 dev, one mid-life flight each)", loc="left"
    )
    fig.tight_layout()
    save(fig, out)
    return info


def fig_sensors_over_life(store: UnitStore, out: Path) -> dict[str, Any]:
    """Raw per-cycle cruise medians of four sensors for one unit."""
    names = store.var_names["X_s"]
    cols = [names.index(s) for s in RAW_SENSORS]
    ua = store.load(EXAMPLE_UNIT)
    cyc, med = per_cycle_cruise_median(store, EXAMPLE_UNIT, ua.X_s, cols)
    on = onset_cycle(store, EXAMPLE_UNIT)
    fig, axes = plt.subplots(2, 2, figsize=(8, 4.8), sharex=True)
    for ax, j, s in zip(axes.flat, range(len(cols)), RAW_SENSORS, strict=True):
        ax.plot(cyc, med[:, j], color=SERIES[0], lw=1.6)
        ax.axvline(on, color=NEUTRAL, lw=1, ls="--")
        ax.set_title(s, loc="left")
    axes[0, 0].annotate(
        "hs onset",
        (on, axes[0, 0].get_ylim()[1]),
        xytext=(4, -10),
        textcoords="offset points",
        color=TEXT_2,
        fontsize=8,
    )
    for ax in axes[1]:
        ax.set_xlabel("cycle")
    fig.suptitle(
        f"Raw sensors over life, DS02 unit {EXAMPLE_UNIT} (median over each flight's cruise run)",
        x=0.01,
        ha="left",
        fontsize=10,
    )
    fig.tight_layout()
    save(fig, out)
    return {"unit": EXAMPLE_UNIT, "onset": on, "n_cycles": len(cyc)}


def fig_hs_onset(store: UnitStore, out: Path) -> dict[str, Any]:
    """Healthy vs abnormal cycles per dev unit."""
    info = {}
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for i, u in enumerate(store.units):
        last = int(store.load(u).flights[-1, 0])
        on = onset_cycle(store, u)
        ax.barh(
            i,
            on - 1,
            left=0.5,
            color=SERIES[0],
            height=0.6,
            label="healthy (hs = 1)" if i == 0 else None,
        )
        ax.barh(
            i,
            last - on + 1,
            left=on - 0.5,
            color=SERIES[1],
            height=0.6,
            label="abnormal (hs = 0)" if i == 0 else None,
        )
        ax.text(
            last + 1, i, f"{last - on + 1} abnormal / {last}", va="center", fontsize=8, color=TEXT_2
        )
        info[str(u)] = {"onset": on, "last_cycle": last, "n_abnormal": last - on + 1}
    ax.set_yticks(range(len(store.units)), [f"unit {u}" for u in store.units])
    ax.set_xlabel("cycle")
    ax.set_xlim(0, max(v["last_cycle"] for v in info.values()) + 18)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2)
    ax.set_title("Health-state onset per DS02 dev unit", loc="left", pad=18)
    fig.tight_layout()
    save(fig, out)
    return info


def fig_residuals(store: UnitStore, out: Path) -> dict[str, Any]:
    """Per-cycle cruise-median residuals of four sensors, all dev units, aligned at onset."""
    names = store.var_names["X_s"]
    cols = [names.index(s) for s in RESIDUAL_SENSORS]
    fig, axes = plt.subplots(2, 2, figsize=(8, 5), sharex=True)
    drift: dict[str, Any] = {s: {} for s in RESIDUAL_SENSORS}
    for k, u in enumerate(store.units):
        res = np.load(store.unit_dir(u) / "residuals.npy", mmap_mode="r")
        cyc, med = per_cycle_cruise_median(store, u, res, cols)
        on = onset_cycle(store, u)
        for ax, j, s in zip(axes.flat, range(len(cols)), RESIDUAL_SENSORS, strict=True):
            ax.plot(cyc - on, med[:, j], color=SERIES[k], lw=1.2, label=f"unit {u}")
            early = np.nanmean(med[(cyc >= 6) & (cyc < on), j])
            late = np.nanmean(med[cyc > cyc.max() - 5, j])
            drift[s][str(u)] = {"pre_onset_mean": float(early), "last5_mean": float(late)}
    for ax, s in zip(axes.flat, RESIDUAL_SENSORS, strict=True):
        ax.axvline(0, color=NEUTRAL, lw=1, ls="--")
        ax.axhline(0, color=NEUTRAL, lw=0.8)
        ax.set_title(f"residual {s}", loc="left")
    for ax in axes[1]:
        ax.set_xlabel("cycles since hs onset")
    for ax in axes[:, 0]:
        ax.set_ylabel("(X_s − f_u(W)) / σ_u")
    axes[0, 1].legend(loc="upper left", ncol=2, fontsize=7)
    fig.suptitle(
        "Healthy-baseline residuals over life (DS02 dev, per-cycle cruise median)",
        x=0.01,
        ha="left",
        fontsize=10,
    )
    fig.tight_layout()
    save(fig, out)
    return drift


def fig_altitude(
    dev: UnitStore, test_w: dict[int, NDArray[np.float32]], out: Path
) -> dict[str, Any]:
    """Altitude histograms: pooled dev units vs test units 14 and 15.

    ``test_w`` holds only the test units' flight descriptors (W, every 10th row), from
    :func:`aerosentinel.data.ncmapss.flight_descriptors_of_test_units`.
    """
    col = dev.var_names["W"].index("alt")
    dev_alt = np.concatenate([np.asarray(dev.load(u).W[::10, col]) for u in dev.units])
    groups = {"dev units (pooled)": dev_alt}
    for u in (14, 15):
        groups[f"test unit {u}"] = test_w[u][:, col]
    lo = min(float(g.min()) for g in groups.values())
    hi = max(float(g.max()) for g in groups.values())
    bins = np.linspace(lo, hi, 60)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    info = {}
    for k, (name, g) in enumerate(groups.items()):
        ax.hist(g, bins=bins, density=True, histtype="step", lw=2, color=SERIES[k], label=name)
        info[name] = {"median_ft": float(np.median(g)), "p95_ft": float(np.quantile(g, 0.95))}
    ax.set_xlabel("altitude [ft]")
    ax.set_ylabel("density")
    ax.legend(loc="upper left")
    ax.set_title("Altitude distribution: DS02 dev units vs test units 14 and 15", loc="left")
    fig.tight_layout()
    save(fig, out)
    return info


def main() -> int:
    """Draw every figure and write the EDA numbers."""
    apply_style()
    paths = get_paths()
    fig_dir = paths.figures
    fig_dir.mkdir(parents=True, exist_ok=True)
    root = paths.data_processed / "ncmapss"
    dev = UnitStore(root, "DS02", "dev")
    test_w = flight_descriptors_of_test_units(root, "DS02", [14, 15], step=10)  # W only
    doc = {
        "experiment": "ncmapss-ds02_data_eda",
        "dataset": "ncmapss-ds02",
        "model": "data",
        "setting": "eda",
        "written_by": "scripts/make_eda_figures.py",
        "flight_profiles": fig_flight_profiles(
            paths.data_raw / "ncmapss", fig_dir / "ncmapss_flight_profiles.png"
        ),
        "sensors_over_life": fig_sensors_over_life(dev, fig_dir / "ncmapss_sensors_over_life.png"),
        "hs_onset": fig_hs_onset(dev, fig_dir / "ncmapss_hs_onset.png"),
        "residual_drift": fig_residuals(dev, fig_dir / "ncmapss_residuals_over_life.png"),
        "altitude": fig_altitude(dev, test_w, fig_dir / "ncmapss_altitude_dev_vs_test.png"),
        "git": git_info(paths.root),
    }
    out = paths.results / f"{doc['experiment']}.json"
    out.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"figures -> {fig_dir}; numbers -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
