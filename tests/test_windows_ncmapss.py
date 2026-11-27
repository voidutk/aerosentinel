"""N-CMAPSS window index (SPEC §1.3 item 3, §11 test_windows): flights, cruise runs, strides."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from aerosentinel.data import ncmapss as nc
from aerosentinel.data import ncmapss_windows as nw
from aerosentinel.data.masks import DEFAULT_MASK, CruiseMaskParams
from aerosentinel.data.ncmapss_derive import compute_cruise_runs
from aerosentinel.data.windows import (
    aggregate_per_cycle,
    build_flight_index,
    n_samples_downsampled,
    n_windows,
)
from synth_ncmapss import write_synthetic_ncmapss

ROOT = Path(__file__).resolve().parents[1]
MASK = CruiseMaskParams(min_len=128)  # synthetic flights are short


@pytest.fixture
def store(tmp_path: Path) -> nc.UnitStore:
    h5 = write_synthetic_ncmapss(tmp_path / "x.h5", [2, 5, 10], [11], rows=600)
    nc.convert_split(h5, "dev", tmp_path / "p" / "DS02" / "dev")
    s = nc.UnitStore(tmp_path / "p", "DS02", "dev")
    compute_cruise_runs(s, MASK)
    return s


@pytest.mark.parametrize(("n", "t", "s", "k"), [(100, 10, 3, 1), (100, 10, 1, 4), (9, 10, 1, 1)])
def test_count_per_segment_and_end_alignment(n: int, t: int, s: int, k: int) -> None:
    idx = build_flight_index([(7, 3, 50, 50 + n)], t, s, k)
    m = n_samples_downsampled(n, k)
    assert len(idx) == n_windows(m, t, s) == (0 if m < t else (m - t) // s + 1)
    if len(idx):
        assert idx.end[-1] == 50 + n - 1  # last window ends on the segment's last row
        assert idx.start.min() >= 50
        assert np.all(np.diff(idx.start) == s * k)
        rows = idx.rows(slice(None))
        assert np.all(np.diff(rows, axis=1) == k)


def test_windows_stay_inside_their_segment() -> None:
    segs = [(1, 1, 0, 40), (1, 2, 40, 95), (2, 1, 0, 33)]
    idx = build_flight_index(segs, 8, 2, 2)
    rows = idx.rows(slice(None))
    for u, c, lo_hi in zip(idx.unit, idx.cycle, rows, strict=True):
        (s, e) = next((s, e) for uu, cc, s, e in segs if (uu, cc) == (u, c))
        assert lo_hi.min() >= s and lo_hi.max() < e


def test_full_flight_windows_never_cross_cycles(store: nc.UnitStore) -> None:
    idx = nw.window_index(
        store, store.units, "full_flight", MASK, window=32, stride=5, downsample=3
    )
    cyc_col = store.a_col("cycle")
    for u in store.units:
        a = store.load(u).A
        sel = idx.unit == u
        rows = idx.rows(np.flatnonzero(sel))
        cyc = a[rows, cyc_col]
        assert np.all(cyc == cyc[:, :1])  # one cycle per window
        assert np.all(cyc[:, 0] == idx.cycle[sel])


def test_cruise_windows_stay_inside_the_cruise_run(store: nc.UnitStore) -> None:
    idx = nw.window_index(
        store, store.units, "cruise_only", MASK, window=32, stride=5, downsample=2
    )
    assert len(idx) > 0
    from aerosentinel.data.ncmapss_derive import load_cruise_runs

    for u in store.units:
        runs = load_cruise_runs(store, u, MASK)
        fl = store.load(u).flights
        sel = np.flatnonzero(idx.unit == u)
        for i in sel:
            j = int(np.flatnonzero(fl[:, 0] == idx.cycle[i])[0])
            s, e = runs[j]
            assert s <= idx.start[i] and idx.end[i] < e


def test_label_is_read_at_window_end(store: nc.UnitStore) -> None:
    idx = nw.window_index(
        store, store.units, "full_flight", MASK, window=16, stride=7, downsample=1
    )
    got = np.concatenate([b.y for b in nw.iter_batches(store, idx, ["W", "X_s"], batch_size=50)])
    want = np.concatenate(
        [store.load(int(u)).Y[[e]] for u, e in zip(idx.unit, idx.end, strict=True)]
    )
    np.testing.assert_array_equal(got, want)


def test_batches_match_direct_slicing_and_channels(store: nc.UnitStore) -> None:
    idx = nw.window_index(
        store, store.units, "full_flight", MASK, window=8, stride=50, downsample=2
    )
    b = next(nw.iter_batches(store, idx, ["W", "X_s"], batch_size=10**6))
    assert b.x.shape == (len(idx), 8, 18)
    u0, s0 = int(idx.unit[0]), int(idx.start[0])
    ua = store.load(u0)
    np.testing.assert_array_equal(b.x[0, :, :4], ua.W[s0 : s0 + 16 : 2])
    assert nw.channel_names(store, ["W", "X_s"])[:4] == ["alt", "Mach", "TRA", "T2"]
    with pytest.raises(ValueError):
        nw.channel_names(store, ["X_v"])


def test_min_cycle_drops_baseline_cycles(store: nc.UnitStore) -> None:
    idx = nw.window_index(
        store, store.units, "full_flight", MASK, window=8, stride=20, downsample=1, min_cycle=6
    )
    assert idx.cycle.min() == 6


def test_index_cache_round_trip(store: nc.UnitStore, tmp_path: Path) -> None:
    kw = {"window": 16, "stride": 4, "downsample": 2, "cache_dir": tmp_path / "idx"}
    a = nw.window_index(store, store.units, "cruise_only", MASK, **kw)  # type: ignore[arg-type]
    files = list((tmp_path / "idx").glob("*.npz"))
    assert len(files) == 1
    b = nw.window_index(store, store.units, "cruise_only", MASK, **kw)  # type: ignore[arg-type]
    assert np.array_equal(a.start, b.start) and (a.window, a.stride, a.downsample) == (16, 4, 2)
    # a different stride is a different key: no stale index is reused
    nw.window_index(store, store.units, "cruise_only", MASK, **{**kw, "stride": 5})  # type: ignore[arg-type]
    assert len(list((tmp_path / "idx").glob("*.npz"))) == 2


def test_index_holds_integers_only(store: nc.UnitStore) -> None:
    idx = nw.window_index(store, store.units, "full_flight", MASK, window=8, stride=1, downsample=1)
    for arr in idx.to_npz().values():
        assert arr.dtype == np.int64 and arr.ndim == 1


def test_coverage_counts_cycles_with_a_window(store: nc.UnitStore) -> None:
    cov = nw.cycle_coverage(store, store.units, "full_flight", MASK, window=10**6, downsample=1)
    assert cov["n_with_window"] == 0 and cov["n_cycles"] > 0
    cov = nw.cycle_coverage(store, store.units, "full_flight", MASK, window=4, downsample=1)
    assert cov["fraction"] == 1.0
    cru = nw.cycle_coverage(store, store.units, "cruise_only", MASK, window=64, downsample=1)
    assert cru["n_with_window"] < cru["n_cycles"]  # the short cycle-2 flights have no cruise run


def test_per_cycle_prediction_is_median_of_windows() -> None:
    u, c, med = aggregate_per_cycle(
        np.array([1.0, 5.0, 3.0, 10.0, 0.0]), np.array([2, 2, 2, 2, 3]), np.array([4, 4, 4, 5, 1])
    )
    assert u.tolist() == [2, 2, 3] and c.tolist() == [4, 5, 1]
    assert med.tolist() == [3.0, 10.0, 0.0]


def test_one_stride_config_key_for_all_splits() -> None:
    """There is no separate test-stride setting anywhere in configs or the pipeline."""
    import yaml

    for p in (ROOT / "configs" / "data").glob("ncmapss_*.yaml"):
        keys = set(yaml.safe_load(p.read_text()))
        assert "stride" in keys
        assert not {k for k in keys if "stride" in k and k != "stride"}
    src = (ROOT / "src" / "aerosentinel" / "pipelines" / "ncmapss.py").read_text()
    names = {n.attr for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Attribute)}
    assert not {n for n in names if "stride" in n and n != "stride"}


def test_default_mask_is_spec_min_len() -> None:
    assert DEFAULT_MASK.min_len == 512 and DEFAULT_MASK.std_max == 50.0
