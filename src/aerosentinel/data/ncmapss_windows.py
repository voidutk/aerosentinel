"""Window indices and on-the-fly batches over an N-CMAPSS :class:`UnitStore`.

Settings (SPEC §1.3 item 4):

- ``full_flight``: each flight (one (unit, cycle)) is one segment.
- ``cruise_only``: each flight's single longest cruise run is one segment; flights without a
  run contribute nothing. Non-contiguous cruise samples are never spliced.

Windows are built by :func:`aerosentinel.data.windows.build_flight_index` inside segments, so
they never cross a (unit, cycle) boundary or a cruise-run edge. The index is cached as a small
``.npz`` of integers keyed by a config hash; windows themselves are never stored.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from aerosentinel.data.masks import CruiseMaskParams
from aerosentinel.data.ncmapss import UnitStore
from aerosentinel.data.ncmapss_derive import load_cruise_runs
from aerosentinel.data.windows import FlightWindowIndex, build_flight_index, n_samples_downsampled

SETTINGS: tuple[str, ...] = ("full_flight", "cruise_only")
#: Channel groups a model may see: W, X_s and the healthy-baseline residuals of X_s.
CHANNEL_GROUPS: tuple[str, ...] = ("W", "X_s", "residuals")


def unit_segments(
    store: UnitStore,
    unit: int,
    setting: str,
    params: CruiseMaskParams,
    min_cycle: int = 1,
) -> list[tuple[int, int, int, int]]:
    """``(unit, cycle, start, end)`` segments of one unit for a setting.

    Parameters
    ----------
    store : UnitStore
        Converted subset/split.
    unit : int
        Unit id.
    setting : {"full_flight", "cruise_only"}
        Flight-stage setting.
    params : CruiseMaskParams
        Cruise mask parameters (``cruise_only``).
    min_cycle : int, default 1
        Drop flights with a smaller cycle number (6 for residual models).
    """
    if setting not in SETTINGS:
        raise ValueError(f"setting must be one of {SETTINGS}")
    flights = np.load(store.unit_dir(unit) / "flights.npy")
    full = setting == "full_flight"
    spans = flights[:, 1:3] if full else load_cruise_runs(store, unit, params)
    out = []
    for (cyc, _, _), (s, e) in zip(flights, spans, strict=True):
        if cyc >= min_cycle and s >= 0:
            out.append((unit, int(cyc), int(s), int(e)))
    return out


def store_segments(
    store: UnitStore,
    units: Sequence[int],
    setting: str,
    params: CruiseMaskParams,
    min_cycle: int = 1,
) -> list[tuple[int, int, int, int]]:
    """Segments of several units, in unit order."""
    return [s for u in units for s in unit_segments(store, u, setting, params, min_cycle)]


def cycle_coverage(
    store: UnitStore,
    units: Sequence[int],
    setting: str,
    params: CruiseMaskParams,
    window: int,
    downsample: int,
    min_cycle: int = 1,
) -> dict[str, Any]:
    """Count the cycles that yield at least one window (``m >= T`` model samples).

    Returns
    -------
    dict
        ``n_cycles`` (flights with cycle >= ``min_cycle``), ``n_with_window``, ``fraction``.
    """
    n_cycles = 0
    for u in units:
        fl = np.load(store.unit_dir(u) / "flights.npy")
        n_cycles += int((fl[:, 0] >= min_cycle).sum())
    segs = store_segments(store, units, setting, params, min_cycle)
    ok = sum(n_samples_downsampled(e - s, downsample) >= window for _, _, s, e in segs)
    return {
        "n_cycles": n_cycles,
        "n_with_window": int(ok),
        "fraction": ok / n_cycles if n_cycles else float("nan"),
    }


def index_key(
    store: UnitStore,
    units: Sequence[int],
    setting: str,
    params: CruiseMaskParams,
    window: int,
    stride: int,
    downsample: int,
    min_cycle: int,
) -> str:
    """Config hash of a window index (cache file name)."""
    d = {
        "dataset": store.dataset,
        "split": store.split,
        "manifest": store.manifest_hash(),
        "units": sorted(int(u) for u in units),
        "setting": setting,
        "mask": params.to_dict(),
        "window": window,
        "stride": stride,
        "downsample": downsample,
        "min_cycle": min_cycle,
    }
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


def window_index(
    store: UnitStore,
    units: Sequence[int],
    setting: str,
    params: CruiseMaskParams,
    *,
    window: int,
    stride: int,
    downsample: int,
    min_cycle: int = 1,
    cache_dir: Path | None = None,
) -> FlightWindowIndex:
    """Build (or load from ``cache_dir``) the window index of some units.

    One ``stride`` serves every split: callers pass the same config value for train,
    validation, calibration and test.
    """
    path = None
    if cache_dir is not None:
        key = index_key(store, units, setting, params, window, stride, downsample, min_cycle)
        path = cache_dir / f"{key}.npz"
        if path.is_file():
            with np.load(path) as z:
                return FlightWindowIndex.from_npz({k: z[k] for k in z.files})
    segs = store_segments(store, units, setting, params, min_cycle)
    idx = build_flight_index(segs, window, stride, downsample)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrs = idx.to_npz()
        np.savez(
            path, unit=arrs["unit"], cycle=arrs["cycle"], start=arrs["start"], meta=arrs["meta"]
        )
    return idx


def channel_names(store: UnitStore, groups: Sequence[str]) -> list[str]:
    """Model input channel names: W and X_s variable names, residuals as ``r_<sensor>``."""
    names: list[str] = []
    for g in groups:
        if g not in CHANNEL_GROUPS:
            raise ValueError(f"channel group {g!r} is not an allowed input ({CHANNEL_GROUPS})")
        if g == "residuals":
            names += [f"r_{s}" for s in store.var_names["X_s"]]
        else:
            names += list(store.var_names[g])
    return names


@dataclass
class Batch:
    """One gathered batch: windows, end labels and keys."""

    x: NDArray[np.float32]  # [B, T, C]
    y: NDArray[np.float32]  # raw RUL at window end [B]
    unit: NDArray[np.int64]
    cycle: NDArray[np.int64]


def iter_batches(
    store: UnitStore,
    index: FlightWindowIndex,
    groups: Sequence[str],
    batch_size: int = 4096,
) -> Iterator[Batch]:
    """Gather windows from the per-unit memory maps, one batch at a time.

    Only the current batch is materialized; the index holds integers only.
    """
    if not len(index):
        return
    use_res = "residuals" in groups
    order = np.arange(len(index))
    for b0 in range(0, len(index), batch_size):
        sel = order[b0 : b0 + batch_size]
        xs, ys = [], []
        units = index.unit[sel]
        for u in np.unique(units):
            m = sel[units == u]
            ua = store.load(int(u), residuals=use_res)
            rows = index.rows(m)
            arrays = {"W": ua.W, "X_s": ua.X_s, "residuals": ua.residuals}
            parts = [np.asarray(arrays[g])[rows] for g in groups]
            xs.append(np.concatenate(parts, axis=2))
            ys.append(np.asarray(ua.Y)[index.end[m]])
        # units are contiguous in the index, so concatenation keeps index order
        yield Batch(
            np.concatenate(xs).astype(np.float32, copy=False),
            np.concatenate(ys).astype(np.float32),
            index.unit[sel],
            index.cycle[sel],
        )
