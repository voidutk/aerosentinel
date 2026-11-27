"""Window index over per-segment contiguous arrays (SPEC §1.3 item 3, §12).

A *segment* is the unit of contiguity a window may not cross: a whole engine trajectory for
C-MAPSS (one row per cycle), one (unit, cycle) flight for N-CMAPSS. Windows are never
materialized: the index stores ``(segment, start)`` integers and datasets slice the
per-segment arrays on the fly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class WindowIndex:
    """Integer index of windows: window ``i`` is ``arrays[segment[i]][start[i]:start[i]+T]``.

    Attributes
    ----------
    segment, start : numpy.ndarray
        int64 arrays of equal length.
    window : int
        Window length T.
    stride : int
        Stride s (one config value for every split).
    """

    segment: IntArray
    start: IntArray
    window: int
    stride: int

    def __len__(self) -> int:
        return int(self.segment.shape[0])

    @property
    def end(self) -> IntArray:
        """Index of the last row of each window (where the label is read)."""
        return self.start + self.window - 1


def n_windows(n: int, window: int, stride: int) -> int:
    """Number of windows in a segment: ``floor((N - T) / s) + 1``, or 0 if ``N < T``."""
    if window < 1 or stride < 1:
        raise ValueError("window and stride must be >= 1")
    return 0 if n < window else (n - window) // stride + 1


def build_window_index(segment_lengths: Sequence[int], window: int, stride: int) -> WindowIndex:
    """Build an end-aligned window index that never crosses a segment boundary.

    Starts are laid out backwards from the segment end, so the last window of each segment
    ends exactly on its last row.

    Parameters
    ----------
    segment_lengths : sequence of int
        Row count N of each segment.
    window, stride : int
        Window length T and stride s.

    Returns
    -------
    WindowIndex
        Windows ordered by segment, then by start.
    """
    segs: list[IntArray] = []
    starts: list[IntArray] = []
    for seg, n in enumerate(segment_lengths):
        k = n_windows(int(n), window, stride)
        if k == 0:
            continue
        st = (int(n) - window) - stride * np.arange(k - 1, -1, -1, dtype=np.int64)
        segs.append(np.full(k, seg, dtype=np.int64))
        starts.append(st)
    if not segs:
        empty = np.zeros(0, dtype=np.int64)
        return WindowIndex(empty, empty.copy(), window, stride)
    return WindowIndex(np.concatenate(segs), np.concatenate(starts), window, stride)


def last_window(x: FloatArray, window: int) -> tuple[FloatArray, int]:
    """Return the last ``window`` rows of a segment, edge-padding at the front if short.

    Padding choice (documented in README): a test unit with N < T cycles is padded by
    repeating its first observed cycle ``T - N`` times. Training uses full windows only.

    Parameters
    ----------
    x : numpy.ndarray
        Segment array [N, C].
    window : int
        Window length T.

    Returns
    -------
    (numpy.ndarray, int)
        Window [T, C] and the number of padded rows (0 if none).
    """
    n = x.shape[0]
    if n == 0:
        raise ValueError("empty segment")
    if n >= window:
        return x[n - window :], 0
    pad = window - n
    return np.concatenate([np.repeat(x[:1], pad, axis=0), x], axis=0), pad


def aggregate_per_cycle(
    pred: FloatArray, unit: IntArray, cycle: IntArray
) -> tuple[IntArray, IntArray, FloatArray]:
    """Per-cycle prediction = median of that (unit, cycle)'s window predictions.

    Parameters
    ----------
    pred : numpy.ndarray
        Window predictions [W] (or [W, Q] for quantiles: median per column).
    unit, cycle : numpy.ndarray
        Key of each window [W].

    Returns
    -------
    (units, cycles, medians)
        Unique keys sorted by (unit, cycle) and the median per key.
    """
    keys = np.stack([np.asarray(unit), np.asarray(cycle)], axis=1)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    p = np.asarray(pred, dtype=np.float64)
    order = np.argsort(inv, kind="stable")
    groups = np.split(p[order], np.cumsum(np.bincount(inv, minlength=len(uniq)))[:-1])
    med = np.stack([np.median(g, axis=0) for g in groups])
    return uniq[:, 0], uniq[:, 1], med


class WindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Torch dataset that slices windows from per-segment arrays on the fly.

    Parameters
    ----------
    arrays : sequence of numpy.ndarray
        Per-segment float32 arrays [N, C].
    labels : sequence of numpy.ndarray
        Per-segment labels [N]; the window label is the label at the window end.
    index : WindowIndex
        Window index over ``arrays``.
    """

    def __init__(
        self,
        arrays: Sequence[NDArray[np.float32]],
        labels: Sequence[NDArray[np.float32]],
        index: WindowIndex,
    ) -> None:
        self.arrays = arrays
        self.labels = labels
        self.index = index

    def __len__(self) -> int:
        return len(self.index)

    def window_view(self, i: int) -> NDArray[np.float32]:
        """Return window ``i`` as a NumPy view into the segment array (no copy)."""
        seg, st = int(self.index.segment[i]), int(self.index.start[i])
        return self.arrays[seg][st : st + self.index.window]

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        seg, end = int(self.index.segment[i]), int(self.index.end[i])
        x = torch.from_numpy(self.window_view(i))
        y = torch.tensor(self.labels[seg][end], dtype=torch.float32)
        return x, y


# --- N-CMAPSS: windows over flights inside per-unit arrays ---------------------------------


@dataclass(frozen=True)
class FlightWindowIndex:
    """Window index ``(unit, cycle, start)`` over per-unit native-resolution arrays.

    Window ``i`` covers the rows ``start[i] + k * arange(T)`` of unit ``unit[i]``'s arrays
    (``k`` = ``downsample``), all inside one flight segment. The label is read at
    :attr:`end`. Only integers are stored; windows are gathered on the fly.

    Attributes
    ----------
    unit, cycle, start : numpy.ndarray
        int64 arrays of equal length; ``start`` is a row offset in the unit's arrays.
    window : int
        Window length T in model samples.
    stride : int
        Stride s in model samples (s * k native rows); the same value for every split.
    downsample : int
        Downsample factor k (every k-th native sample).
    """

    unit: IntArray
    cycle: IntArray
    start: IntArray
    window: int
    stride: int
    downsample: int

    def __len__(self) -> int:
        return int(self.unit.shape[0])

    @property
    def end(self) -> IntArray:
        """Row offset of the last sample of each window (where the label is read)."""
        return self.start + self.downsample * (self.window - 1)

    def rows(self, i: slice | IntArray) -> IntArray:
        """Row offsets ``[B, T]`` of the selected windows."""
        st = self.start[i]
        return st[:, None] + self.downsample * np.arange(self.window, dtype=np.int64)[None, :]

    def to_npz(self) -> dict[str, IntArray]:
        """Arrays for :func:`numpy.savez` (inverse of :meth:`from_npz`)."""
        meta = np.array([self.window, self.stride, self.downsample], dtype=np.int64)
        return {"unit": self.unit, "cycle": self.cycle, "start": self.start, "meta": meta}

    @classmethod
    def from_npz(cls, d: Mapping[str, IntArray]) -> FlightWindowIndex:
        """Rebuild an index saved with :meth:`to_npz`."""
        t, s, k = (int(v) for v in d["meta"])
        return cls(np.asarray(d["unit"]), np.asarray(d["cycle"]), np.asarray(d["start"]), t, s, k)


def n_samples_downsampled(length: int, downsample: int) -> int:
    """Model samples in a segment of ``length`` native rows, keeping every k-th row."""
    return 0 if length <= 0 else (length - 1) // downsample + 1


def build_flight_index(
    segments: Sequence[tuple[int, int, int, int]], window: int, stride: int, downsample: int = 1
) -> FlightWindowIndex:
    """End-aligned windows inside each ``(unit, cycle, seg_start, seg_end)`` segment.

    Downsampling keeps native rows ``seg_end - 1 - k*j`` (anchored at the segment end), so the
    last window of every segment ends exactly on the segment's last row. A segment with
    ``m`` model samples yields :func:`n_windows` ``(m, T, s)`` windows; none if ``m < T``.
    Segments never share a window, so windows cannot cross a (unit, cycle) boundary or a
    cruise-run edge.

    Parameters
    ----------
    segments : sequence of (unit, cycle, seg_start, seg_end)
        Half-open native row ranges in each unit's arrays (one per flight, at most).
    window, stride : int
        T and s in model samples.
    downsample : int, default 1
        k.

    Returns
    -------
    FlightWindowIndex
        Windows ordered as the segments, then by start.
    """
    if downsample < 1:
        raise ValueError("downsample must be >= 1")
    us: list[IntArray] = []
    cs: list[IntArray] = []
    sts: list[IntArray] = []
    k = downsample
    for unit, cycle, s0, s1 in segments:
        m = n_samples_downsampled(int(s1) - int(s0), k)
        nw = n_windows(m, window, stride)
        if nw == 0:
            continue
        ends = (int(s1) - 1) - k * stride * np.arange(nw - 1, -1, -1, dtype=np.int64)
        sts.append(ends - k * (window - 1))
        us.append(np.full(nw, unit, dtype=np.int64))
        cs.append(np.full(nw, cycle, dtype=np.int64))
    if not us:
        e = np.zeros(0, dtype=np.int64)
        return FlightWindowIndex(e, e.copy(), e.copy(), window, stride, downsample)
    return FlightWindowIndex(
        np.concatenate(us), np.concatenate(cs), np.concatenate(sts), window, stride, downsample
    )
