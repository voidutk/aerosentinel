"""Cruise-stage mask for N-CMAPSS flights (SPEC §1.3 item 4).

Our own design: for one flight (one (unit, cycle)) at native resolution,

1. smooth altitude with a centred rolling median (``med_win`` samples);
2. reject flights whose smoothed altitude span is below ``min_span_ft`` (no climb/cruise);
3. *high altitude*: smoothed altitude at least ``high_frac`` of the way from the flight's
   minimum to its maximum;
4. *stable*: the smoothed altitude varies by at most ``2 * std_max`` ft (peak to peak) inside
   a centred window of ``stab_win`` samples;
5. candidate runs are maximal contiguous high & stable stretches; a run is accepted if it is
   at least ``min_len`` samples long and the raw altitude std over the run is below
   ``std_max`` ft;
6. the **longest** accepted run is the flight's cruise run, returned as one half-open
   ``[start, end)`` interval in flight-local sample indices.

Windows in the ``cruise_only`` setting lie entirely inside that one run. Non-contiguous
cruise samples are never spliced together.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import maximum_filter1d, median_filter, minimum_filter1d


@dataclass(frozen=True)
class CruiseMaskParams:
    """Parameters of :func:`cruise_run` (part of every window-index cache key)."""

    min_len: int = 512
    std_max: float = 50.0
    med_win: int = 31
    stab_win: int = 61
    high_frac: float = 0.5
    min_span_ft: float = 500.0

    def to_dict(self) -> dict[str, float | int]:
        """Plain dict (for configs, hashes and results JSONs)."""
        return asdict(self)


DEFAULT_MASK = CruiseMaskParams()


def true_runs(mask: ArrayLike) -> list[tuple[int, int]]:
    """Half-open ``[start, end)`` runs of True values in a boolean vector."""
    m = np.asarray(mask, dtype=bool)
    if m.size == 0:
        return []
    edges = np.flatnonzero(np.diff(np.r_[0, m.astype(np.int8), 0]))
    return [(int(a), int(b)) for a, b in zip(edges[0::2], edges[1::2], strict=True)]


def smooth_altitude(alt: ArrayLike, med_win: int) -> NDArray[np.float64]:
    """Centred rolling median of altitude (edges use the nearest sample)."""
    a = np.asarray(alt, dtype=np.float64)
    return np.asarray(median_filter(a, size=med_win, mode="nearest"), dtype=np.float64)


def cruise_run(alt: ArrayLike, params: CruiseMaskParams = DEFAULT_MASK) -> tuple[int, int] | None:
    """Longest stable high-altitude run of one flight, or None.

    Parameters
    ----------
    alt : array-like
        Altitude [ft] of one flight at native resolution.
    params : CruiseMaskParams
        Mask parameters.

    Returns
    -------
    (int, int) or None
        Half-open flight-local ``[start, end)`` of the cruise run, or None if no run of at
        least ``params.min_len`` samples passes the stability test.
    """
    a = np.asarray(alt, dtype=np.float64)
    if a.size < params.min_len:
        return None
    sm = smooth_altitude(a, params.med_win)
    lo, hi = float(sm.min()), float(sm.max())
    if hi - lo < params.min_span_ft:
        return None
    high = sm >= lo + params.high_frac * (hi - lo)
    p2p = maximum_filter1d(sm, params.stab_win, mode="nearest") - minimum_filter1d(
        sm, params.stab_win, mode="nearest"
    )
    stable = p2p <= 2.0 * params.std_max
    best: tuple[int, int] | None = None
    for s, e in true_runs(high & stable):
        if e - s < params.min_len or float(a[s:e].std()) >= params.std_max:
            continue
        if best is None or e - s > best[1] - best[0]:
            best = (s, e)
    return best


def run_mask(n: int, run: tuple[int, int] | None) -> NDArray[np.bool_]:
    """Boolean mask of length ``n`` that is True on ``run`` (all False for None)."""
    m = np.zeros(n, dtype=bool)
    if run is not None:
        m[run[0] : run[1]] = True
    return m


def interval_iou(a: tuple[int, int] | None, b: tuple[int, int] | None) -> float:
    """Intersection over union of two half-open intervals (1.0 if both are None)."""
    if a is None and b is None:
        return 1.0
    if a is None or b is None:
        return 0.0
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union else 1.0
