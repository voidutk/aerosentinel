"""Window index: counts, boundaries, end alignment, labels, aggregation, no materialization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from aerosentinel.data.prepare import prepare_dev
from aerosentinel.data.windows import (
    WindowDataset,
    aggregate_per_cycle,
    build_window_index,
    last_window,
    n_windows,
)


@pytest.mark.parametrize("n", [0, 1, 29, 30, 31, 57, 100])
@pytest.mark.parametrize("t", [1, 10, 30])
@pytest.mark.parametrize("s", [1, 3, 10])
def test_count_formula(n: int, t: int, s: int) -> None:
    expected = 0 if n < t else (n - t) // s + 1
    assert n_windows(n, t, s) == expected
    assert len(build_window_index([n], t, s)) == expected


def test_windows_never_cross_segments_and_end_align() -> None:
    lengths = [57, 12, 30, 101]
    t, s = 30, 7
    idx = build_window_index(lengths, t, s)
    seg_len = np.array(lengths)[idx.segment]
    assert (idx.start >= 0).all() and (idx.end < seg_len).all()
    assert idx.segment.dtype == np.int64 and idx.start.dtype == np.int64
    for seg, n in enumerate(lengths):
        ends = idx.end[idx.segment == seg]
        if n < t:
            assert ends.size == 0  # too-short segment contributes no window
            continue
        assert ends.max() == n - 1  # last window ends on the segment's last row
        assert (np.diff(ends) == s).all()


def test_label_at_window_end_and_views_share_memory() -> None:
    arrays = [np.arange(40 * 3, dtype=np.float32).reshape(40, 3), np.ones((35, 3), np.float32)]
    labels = [np.arange(40, dtype=np.float32)[::-1].copy(), np.arange(35, dtype=np.float32)]
    idx = build_window_index([40, 35], 30, 1)
    ds = WindowDataset(arrays, labels, idx)
    for i in range(len(ds)):
        x, y = ds[i]
        seg, end = int(idx.segment[i]), int(idx.end[i])
        assert x.shape == (30, 3)
        assert float(y) == labels[seg][end]
        # Sliced on the fly from the per-unit array: a view, not a copy.
        assert np.shares_memory(ds.window_view(i), arrays[seg])
        assert np.shares_memory(x.numpy(), arrays[seg])
        assert torch.equal(x, torch.from_numpy(arrays[seg][end - 29 : end + 1]))


def test_last_window_and_padding() -> None:
    x = np.arange(12, dtype=np.float64).reshape(6, 2)
    w, pad = last_window(x, 4)
    assert pad == 0
    np.testing.assert_array_equal(w, x[-4:])
    w, pad = last_window(x, 9)
    assert pad == 3 and w.shape == (9, 2)
    np.testing.assert_array_equal(w[:3], np.repeat(x[:1], 3, axis=0))  # first cycle repeated
    np.testing.assert_array_equal(w[3:], x)


def test_per_cycle_median() -> None:
    pred = np.array([1.0, 3.0, 2.0, 10.0, 20.0, 5.0])
    unit = np.array([1, 1, 1, 1, 2, 2])
    cycle = np.array([1, 1, 1, 2, 1, 1])
    u, c, m = aggregate_per_cycle(pred, unit, cycle)
    np.testing.assert_array_equal(u, [1, 1, 2])
    np.testing.assert_array_equal(c, [1, 2, 1])
    np.testing.assert_allclose(m, [2.0, 10.0, 12.5])
    q = np.stack([pred, pred + 1], axis=1)  # quantiles: median per column
    np.testing.assert_allclose(aggregate_per_cycle(q, unit, cycle)[2][:, 1], [3.0, 11.0, 13.5])


def test_one_stride_for_every_split(synthetic_root: Path) -> None:
    dev = prepare_dev(
        synthetic_root / "data" / "raw" / "cmapss",
        "FD001",
        cap=125,
        window=10,
        stride=3,
        val_frac=0.2,
        split_seed=0,
        n_regimes=1,
        var_threshold=1e-4,
    )
    assert dev.train.index.stride == dev.val.index.stride == 3
    assert dev.train.index.window == dev.val.index.window == 10


def test_batches_match_dataset() -> None:
    arrays = [np.random.default_rng(0).normal(size=(50, 4)).astype(np.float32)]
    from aerosentinel.data.prepare import WindowData

    idx = build_window_index([50], 30, 2)
    data = WindowData(
        [1],
        arrays,
        [np.arange(50, dtype=np.float32)],
        [np.zeros(50, np.int64)],
        idx,
        ["s2", "s3", "s4", "s7"],
    )
    xs, ys = zip(*[(x, y) for x, y, _ in data.batches(batch_size=4)], strict=True)
    ds = WindowDataset(data.arrays, data.labels, idx)
    np.testing.assert_array_equal(
        np.concatenate(xs), np.stack([ds[i][0].numpy() for i in range(len(ds))])
    )
    np.testing.assert_array_equal(np.concatenate(ys), data.window_labels())
