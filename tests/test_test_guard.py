"""Official C-MAPSS test files are readable only with ``allow_test=True`` (SPEC §1.3 item 6).

Only ``aerosentinel.evaluate`` passes the flag; this file uses it on synthetic data to show
that the guarded path still works.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from aerosentinel.data import cmapss as cm
from aerosentinel.data.prepare import prepare_dev, prepare_test

ROOT = Path(__file__).resolve().parents[1]


def _raw(root: Path) -> Path:
    return root / "data" / "raw" / "cmapss"


def test_load_test_requires_allow_flag(synthetic_root: Path) -> None:
    with pytest.raises(cm.TestDataAccessError):
        cm.load_test(_raw(synthetic_root), "FD001")
    df, rul = cm.load_test(_raw(synthetic_root), "FD001", allow_test=True)
    assert df["unit"].nunique() == len(rul) == 12


@pytest.mark.parametrize("name", ["test_FD001.txt", "RUL_FD001.txt"])
def test_low_level_readers_are_guarded_too(synthetic_root: Path, name: str) -> None:
    reader = cm.read_rul_file if name.startswith("RUL_") else cm.read_cmapss_table
    with pytest.raises(cm.TestDataAccessError):
        reader(_raw(synthetic_root) / name)
    reader(_raw(synthetic_root) / name, allow_test=True)
    cm.read_cmapss_table(_raw(synthetic_root) / "train_FD001.txt")  # train is never guarded


def test_prepare_test_requires_allow_flag(synthetic_root: Path) -> None:
    dev = prepare_dev(
        _raw(synthetic_root),
        "FD001",
        cap=125,
        window=10,
        stride=1,
        val_frac=0.2,
        split_seed=0,
        n_regimes=1,
        var_threshold=1e-4,
    )
    with pytest.raises(cm.TestDataAccessError):
        prepare_test(_raw(synthetic_root), "FD001", dev.pre, 10)
    test = prepare_test(_raw(synthetic_root), "FD001", dev.pre, 10, allow_test=True)
    assert len(test.units) == 12


def test_test_unit_ids_must_be_one_to_n(synthetic_root: Path) -> None:
    p = _raw(synthetic_root) / "test_FD001.txt"
    lines = p.read_text().splitlines(keepends=True)
    p.write_text("".join(ln for ln in lines if not ln.startswith("3 ")))  # drop unit 3
    with pytest.raises(ValueError, match="not exactly 1..11"):
        cm.load_test(_raw(synthetic_root), "FD001", allow_test=True)


EVALUATE = "src/aerosentinel/evaluate.py"
#: The only files allowed to open official test data, each with the reason.
TEST_OPENERS = {
    EVALUATE: "C-MAPSS final evaluation",
    "src/aerosentinel/pipelines/ncmapss.py": "N-CMAPSS final evaluation (evaluate_ncmapss)",
    "src/aerosentinel/data/ncmapss.py": "W-only accessor for the EDA altitude figure",
    "scripts/prepare_ncmapss.py": (
        "label-free per-unit derivations: cruise runs (W) and healthy-baseline residuals "
        "(W, X_s and the unit's own cycles 1-5); never Y or hs"
    ),
}
#: Functions that open test data, and the only files allowed to call them.
GUARDED_CALLS = {
    "evaluate_ncmapss": {EVALUATE},
    "flight_descriptors_of_test_units": {"scripts/make_eda_figures.py"},
}


def _code_files() -> list[Path]:
    dirs = [ROOT / d for d in ("src", "scripts", "app") if (ROOT / d).is_dir()]
    return sorted(p for d in dirs for p in d.rglob("*.py"))


def _allow_test_keywords(path: Path) -> list[ast.expr]:
    """Values passed as ``allow_test=...`` in every call in ``path``."""
    tree = ast.parse(path.read_text())
    return [
        kw.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for kw in node.keywords
        if kw.arg == "allow_test"
    ]


def _is_literal_true(v: ast.expr) -> bool:
    return isinstance(v, ast.Constant) and v.value is True


def _is_pass_through(v: ast.expr) -> bool:
    return isinstance(v, ast.Name) and v.id == "allow_test"


def test_only_evaluate_passes_the_allow_flag() -> None:
    """Only the files in ``TEST_OPENERS`` may pass ``allow_test=True``.

    Every other ``allow_test=...`` keyword in src/, scripts/ and app/ must forward the
    caller's own ``allow_test`` parameter unchanged (the loader chain in cmapss.py and
    prepare.py); a variable, expression or literal True anywhere else fails. The string
    ``"allow_test"`` (e.g. a ``**{"allow_test": True}`` splat) is banned outside them.

    This is a guard against accidental use, not a sandbox: code that reads the files with
    pandas or h5py directly is not caught here.
    """
    users = []
    for p in _code_files():
        rel = str(p.relative_to(ROOT))
        values = _allow_test_keywords(p)
        if any(_is_literal_true(v) for v in values):
            users.append(rel)
        if rel not in TEST_OPENERS:
            bad = [ast.unparse(v) for v in values if not _is_pass_through(v)]
            assert not bad, f"{rel} passes allow_test={bad}"
            assert '"allow_test"' not in p.read_text() and "'allow_test'" not in p.read_text(), rel
    assert sorted(users) == sorted(TEST_OPENERS), users


def _called_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            out.add(f.id if isinstance(f, ast.Name) else getattr(f, "attr", ""))
    return out


@pytest.mark.parametrize("func", sorted(GUARDED_CALLS))
def test_test_openers_have_only_their_documented_callers(func: str) -> None:
    callers = {str(p.relative_to(ROOT)) for p in _code_files() if func in _called_names(p)}
    assert callers == GUARDED_CALLS[func]


def test_eda_test_accessor_returns_flight_descriptors_only(tmp_path: Path) -> None:
    from aerosentinel.data import ncmapss as nc
    from synth_ncmapss import write_synthetic_ncmapss

    h5 = write_synthetic_ncmapss(tmp_path / "N-CMAPSS_DS02-006.h5", [2], [11, 14], rows=50)
    root = tmp_path / "ncmapss"
    nc.convert_split(h5, "test", root / "DS02" / "test")
    out = nc.flight_descriptors_of_test_units(root, "DS02", [14], step=5)
    assert list(out) == [14]
    store = nc.UnitStore(root, "DS02", "test", allow_test=True)
    np.testing.assert_array_equal(out[14], store.load(14).W[::5])
    assert not isinstance(out[14], np.memmap)  # a copy, no handle into the store


def test_derivations_never_read_labels_or_health_state(tmp_path: Path) -> None:
    """Cruise runs and residuals of test units read neither Y nor the hs/Fc columns of A."""
    from aerosentinel.data import ncmapss as nc
    from aerosentinel.data.baseline_residuals import BaselineConfig
    from aerosentinel.data.masks import CruiseMaskParams
    from aerosentinel.data.ncmapss_derive import (
        compute_cruise_runs,
        compute_residuals,
        mask_hash,
    )
    from synth_ncmapss import write_synthetic_ncmapss

    h5 = write_synthetic_ncmapss(tmp_path / "N-CMAPSS_DS02-006.h5", [2], [11], rows=300)
    root = tmp_path / "ncmapss"
    nc.convert_split(h5, "test", root / "DS02" / "test")
    store = nc.UnitStore(root, "DS02", "test", allow_test=True)
    ref = compute_residuals(store, BaselineConfig(steps=20, hidden=8))
    ud = store.unit_dir(11)
    # Poison the label and health-state columns: the derivations must not change.
    y = np.load(ud / "Y.npy")
    np.save(ud / "Y.npy", np.full_like(y, np.nan))
    a = np.load(ud / "A.npy")
    for col in ("hs", "Fc"):
        a[:, store.a_col(col)] = np.nan
    np.save(ud / "A.npy", a)
    mask = CruiseMaskParams(min_len=64)
    r_ref = np.load(ud / "residuals.npy").copy()
    compute_cruise_runs(store, mask)
    runs_ref = np.load(ud / f"cruise_{mask_hash(mask)}.npy").copy()
    again = compute_residuals(store, BaselineConfig(steps=20, hidden=8))
    compute_cruise_runs(store, mask)
    np.testing.assert_array_equal(np.load(ud / "residuals.npy"), r_ref)
    np.testing.assert_array_equal(np.load(ud / f"cruise_{mask_hash(mask)}.npy"), runs_ref)
    assert again.keys() == ref.keys()
