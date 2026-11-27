"""N-CMAPSS conversion to per-unit arrays and the test-split guard (synthetic .h5)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from aerosentinel.data import ncmapss as nc
from synth_ncmapss import write_synthetic_ncmapss


@pytest.fixture
def h5_file(tmp_path: Path) -> Path:
    return write_synthetic_ncmapss(tmp_path / "N-CMAPSS_DS02-006.h5", [2, 5, 10], [11], rows=300)


def test_describe_lists_keys_and_names(h5_file: Path) -> None:
    info = nc.describe_h5(h5_file)
    assert {"W_dev", "X_s_test", "A_var", "Y_dev"} <= set(info["datasets"])
    assert info["var_names"]["A"] == ["unit", "cycle", "Fc", "hs"]
    assert len(info["var_names"]["X_s"]) == 14


@pytest.mark.parametrize("chunk_rows", [1000, 10**7])  # chunk boundaries mid-unit / one read
def test_convert_round_trip(h5_file: Path, tmp_path: Path, chunk_rows: int) -> None:
    out = tmp_path / "proc" / "DS02" / "dev"
    man = nc.convert_split(h5_file, "dev", out, chunk_rows=chunk_rows)
    assert sorted(man["units"]) == ["10", "2", "5"]
    assert set(man["var_names"]) == {"W", "X_s", "A", "Y"}  # X_v and T are never stored
    store = nc.UnitStore(tmp_path / "proc", "DS02", "dev")
    with h5py.File(h5_file) as f:
        a = f["A_dev"][:]
        for u in store.units:
            m = a[:, 0] == u
            ua = store.load(u)
            np.testing.assert_allclose(ua.W, f["W_dev"][:][m].astype(np.float32))
            np.testing.assert_allclose(ua.X_s, f["X_s_dev"][:][m].astype(np.float32))
            np.testing.assert_array_equal(ua.Y, f["Y_dev"][:][m, 0].astype(np.float32))
            assert ua.W.dtype == np.float32 and not ua.W.flags.writeable
            fl = ua.flights
            assert fl[0, 1] == 0 and fl[-1, 2] == len(ua.W)
            assert np.all(fl[1:, 1] == fl[:-1, 2])  # flights tile the unit's rows
            for cyc, s, e in fl:
                assert np.all(ua.A[s:e, store.a_col("cycle")] == cyc)
    assert not list(out.rglob("X_v.npy")) and not list(out.rglob("T.npy"))


def test_test_split_requires_allow_flag(h5_file: Path, tmp_path: Path) -> None:
    nc.convert_split(h5_file, "test", tmp_path / "p" / "DS02" / "test")
    with pytest.raises(nc.TestDataAccessError):
        nc.UnitStore(tmp_path / "p", "DS02", "test")
    assert nc.UnitStore(tmp_path / "p", "DS02", "test", allow_test=True).units == [11]


def test_noncontiguous_units_and_decreasing_cycles_rejected() -> None:
    with pytest.raises(ValueError):
        nc.unit_row_ranges(np.array([1, 1, 2, 1]))
    with pytest.raises(ValueError):
        nc.flight_table(np.array([1, 2, 1]))
    assert nc.unit_row_ranges(np.array([3, 3, 4])) == {3: (0, 2), 4: (2, 3)}
    np.testing.assert_array_equal(
        nc.flight_table(np.array([1, 1, 2, 2, 2])), [[1, 0, 2], [2, 2, 5]]
    )
