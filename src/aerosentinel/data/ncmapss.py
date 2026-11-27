"""N-CMAPSS HDF5 access: key inspection, chunked conversion to per-unit arrays, loading.

SPEC §1.2: the inputs are W (4 flight descriptors) and X_s (14 measured sensors). Y (RUL)
and A (unit, cycle, Fc, hs) are kept as labels / metadata only. X_v (virtual sensors) and
T (theta, health parameters) are not converted.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numpy.typing import NDArray

H5_GROUPS: tuple[str, ...] = ("W", "X_s", "X_v", "T", "Y", "A")


def decode_names(values: Any) -> list[str]:
    """Decode an HDF5 ``*_var`` name array (bytes or str) into Python strings."""
    out = []
    for v in np.asarray(values).reshape(-1):
        out.append(v.decode() if isinstance(v, bytes | np.bytes_) else str(v))
    return out


def describe_h5(path: Path) -> dict[str, Any]:
    """Keys, shapes, dtypes, chunking and decoded variable names of an N-CMAPSS file.

    Parameters
    ----------
    path : pathlib.Path
        ``.h5`` file.

    Returns
    -------
    dict
        ``{"datasets": {key: {"shape", "dtype", "chunks", "compression"}},
        "var_names": {group: [names]}}``.
    """
    out: dict[str, Any] = {"datasets": {}, "var_names": {}}
    with h5py.File(path, "r") as f:
        for key in sorted(f.keys()):
            d = f[key]
            out["datasets"][key] = {
                "shape": list(d.shape),
                "dtype": str(d.dtype),
                "chunks": list(d.chunks) if d.chunks else None,
                "compression": d.compression,
            }
            if key.endswith("_var"):
                out["var_names"][key.removesuffix("_var")] = decode_names(d[()])
    return out


# --- conversion to per-unit arrays ---------------------------------------------------------

#: Groups written to the per-unit store. X_v and T (theta) are never converted.
STORE_GROUPS: tuple[str, ...] = ("W", "X_s", "Y", "A")
SPLITS: tuple[str, ...] = ("dev", "test")
STORE_VERSION = 1


class TestDataAccessError(PermissionError):
    """Official N-CMAPSS test units were requested without ``allow_test=True``."""

    __test__ = False  # not a pytest test class


def unit_row_ranges(unit_col: NDArray[Any]) -> dict[int, tuple[int, int]]:
    """Half-open row range of each unit, requiring each unit's rows to be contiguous.

    Raises
    ------
    ValueError
        If a unit appears in more than one contiguous block.
    """
    u = np.asarray(unit_col).astype(np.int64)
    if u.size == 0:
        return {}
    starts = np.r_[0, np.flatnonzero(np.diff(u)) + 1]
    ends = np.r_[starts[1:], u.size]
    out: dict[int, tuple[int, int]] = {}
    for s, e in zip(starts, ends, strict=True):
        key = int(u[s])
        if key in out:
            raise ValueError(f"unit {key} rows are not contiguous")
        out[key] = (int(s), int(e))
    return out


def flight_table(cycle: NDArray[Any]) -> NDArray[np.int64]:
    """``[n_flights, 3]`` table of ``(cycle, start, end)`` row ranges within one unit.

    Raises
    ------
    ValueError
        If cycles decrease or a cycle appears in two separate blocks.
    """
    c = np.asarray(cycle).astype(np.int64)
    if np.any(np.diff(c) < 0):
        raise ValueError("cycles must be non-decreasing within a unit")
    starts = np.r_[0, np.flatnonzero(np.diff(c)) + 1]
    ends = np.r_[starts[1:], c.size]
    return np.stack([c[starts], starts, ends], axis=1).astype(np.int64)


def convert_split(
    h5_path: Path,
    split: str,
    out_dir: Path,
    *,
    chunk_rows: int = 1_000_000,
    source_sha256: str | None = None,
) -> dict[str, Any]:
    """Convert one split of an N-CMAPSS file to per-unit float32 ``.npy`` arrays.

    Two chunked passes over the HDF5 file: pass 1 reads the ``unit`` and ``cycle`` columns
    of ``A`` (checking contiguity and cycle order); pass 2 copies each unit's rows of W,
    X_s, Y and A into ``out_dir/unit_<u>/<group>.npy`` (memory-mapped writes, at most
    ``chunk_rows`` rows in memory per group) and writes ``flights.npy``.

    Parameters
    ----------
    h5_path : pathlib.Path
        Source ``.h5``.
    split : {"dev", "test"}
        Split suffix.
    out_dir : pathlib.Path
        Destination ``<root>/<DS>/<split>``.
    chunk_rows : int, default 1_000_000
        Rows per HDF5 read.
    source_sha256 : str or None
        Recorded in the manifest.

    Returns
    -------
    dict
        The manifest written to ``out_dir/manifest.json``.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "r") as f:
        names = {g: decode_names(f[f"{g}_var"][()]) for g in ("W", "X_s", "A")}
        names["Y"] = ["RUL"]
        a_key = f"A_{split}"
        n = int(f[a_key].shape[0])
        iu, ic = names["A"].index("unit"), names["A"].index("cycle")
        ucol = np.empty(n, dtype=np.int64)
        ccol = np.empty(n, dtype=np.int64)
        for a in range(0, n, chunk_rows):  # pass 1
            blk = f[a_key][a : a + chunk_rows]
            ucol[a : a + len(blk)] = blk[:, iu]
            ccol[a : a + len(blk)] = blk[:, ic]
        ranges = unit_row_ranges(ucol)
        units_meta: dict[str, Any] = {}
        for u, (s, e) in sorted(ranges.items()):  # pass 2
            ud = out_dir / f"unit_{u}"
            ud.mkdir(exist_ok=True)
            flights = flight_table(ccol[s:e])
            np.save(ud / "flights.npy", flights)
            for g in STORE_GROUPS:
                ds = f[f"{g}_{split}"]
                mm = np.lib.format.open_memmap(
                    ud / f"{g}.npy", mode="w+", dtype=np.float32, shape=(e - s, ds.shape[1])
                )
                for a in range(s, e, chunk_rows):
                    b = min(a + chunk_rows, e)
                    mm[a - s : b - s] = ds[a:b].astype(np.float32)
                mm.flush()
                del mm
            units_meta[str(u)] = {"rows": e - s, "n_flights": int(len(flights))}
    manifest = {
        "version": STORE_VERSION,
        "source": h5_path.name,
        "source_sha256": source_sha256,
        "split": split,
        "var_names": names,
        "n_rows": n,
        "units": units_meta,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


# --- loading -------------------------------------------------------------------------------


@dataclass
class UnitArrays:
    """Read-only memory maps of one unit (native resolution, float32)."""

    unit: int
    W: NDArray[np.float32]
    X_s: NDArray[np.float32]
    Y: NDArray[np.float32]
    A: NDArray[np.float32]
    flights: NDArray[np.int64]
    residuals: NDArray[np.float32] | None


class UnitStore:
    """Per-unit arrays of one N-CMAPSS subset and split.

    Parameters
    ----------
    root : pathlib.Path
        ``data/processed/ncmapss``.
    dataset : str
        Subset id, e.g. ``"DS02"``.
    split : {"dev", "test"}
        Split.
    allow_test : bool, default False
        Required for ``split="test"``. Only the final evaluation (and the documented
        W-only EDA figure) passes it.
    """

    def __init__(self, root: Path, dataset: str, split: str, *, allow_test: bool = False):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        if split == "test" and not allow_test:
            raise TestDataAccessError(
                f"{dataset} test units are touched only by the final evaluation "
                "(pass allow_test=True there and nowhere else)"
            )
        self.dataset = dataset
        self.split = split
        self.dir = root / dataset / split
        manifest = self.dir / "manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError(
                f"{manifest} missing: run `uv run python scripts/prepare_ncmapss.py "
                f"--datasets {dataset}`"
            )
        self.manifest: dict[str, Any] = json.loads(manifest.read_text())
        self.units: list[int] = sorted(int(u) for u in self.manifest["units"])
        self.var_names: dict[str, list[str]] = self.manifest["var_names"]

    def a_col(self, name: str) -> int:
        """Column of ``name`` (``unit``, ``cycle``, ``Fc``, ``hs``) in A."""
        return self.var_names["A"].index(name)

    def unit_dir(self, unit: int) -> Path:
        """Directory of one unit's arrays."""
        return self.dir / f"unit_{unit}"

    def load(self, unit: int, *, residuals: bool = False) -> UnitArrays:
        """Memory-map one unit's arrays (read-only)."""
        d = self.unit_dir(unit)

        def mm(name: str) -> NDArray[np.float32]:
            arr: NDArray[np.float32] = np.load(d / f"{name}.npy", mmap_mode="r")
            return arr

        res = mm("residuals") if residuals else None
        return UnitArrays(
            unit=unit,
            W=mm("W"),
            X_s=mm("X_s"),
            Y=mm("Y")[:, 0],
            A=mm("A"),
            flights=np.load(d / "flights.npy"),
            residuals=res,
        )

    def manifest_hash(self) -> str:
        """Short hash of the manifest (part of window-index cache keys)."""
        blob = json.dumps(self.manifest, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


def flight_descriptors_of_test_units(
    root: Path, dataset: str, units: list[int] | None = None, *, step: int = 1
) -> dict[int, NDArray[np.float32]]:
    """W (flight descriptors only) of official test units, for the documented EDA figure.

    The one place outside the final evaluation that opens the test split. Only W leaves
    this function, as an in-memory copy: no sensors, labels (Y) or metadata (A).

    Parameters
    ----------
    root : pathlib.Path
        ``data/processed/ncmapss``.
    dataset : str
        Subset id, e.g. ``"DS02"``.
    units : list of int or None
        Test units to return; all if None.
    step : int, default 1
        Row subsampling step.

    Returns
    -------
    dict
        ``{unit: W[::step]}`` (float32, columns as in ``var_names["W"]``).
    """
    store = UnitStore(root, dataset, "test", allow_test=True)
    wanted = store.units if units is None else units
    return {u: np.array(store.load(u).W[::step]) for u in wanted}
