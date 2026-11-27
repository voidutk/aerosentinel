"""Unit-level splits: disjoint unit sets, no window spanning two units, OOF folds."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aerosentinel.data.prepare import prepare_dev
from aerosentinel.data.splits import train_val_units, unit_kfold


@pytest.mark.parametrize("n", [10, 100, 249])
def test_train_val_disjoint_and_complete(n: int) -> None:
    units = list(range(1, n + 1))
    tr, va = train_val_units(units, 0.2, seed=0)
    assert set(tr).isdisjoint(va)
    assert set(tr) | set(va) == set(units)
    assert len(va) == round(0.2 * n)
    assert train_val_units(units, 0.2, seed=0) == (tr, va)  # deterministic
    assert train_val_units(units, 0.2, seed=1) != (tr, va)


def test_kfold_holds_out_every_unit_exactly_once() -> None:
    units = list(range(1, 261))
    folds = unit_kfold(units, k=5, seed=0)
    held = [u for _, h in folds for u in h]
    assert sorted(held) == units  # every unit held out exactly once
    for train, h in folds:
        # OOF: the fold model that predicts h never trained on any unit of h.
        assert set(train).isdisjoint(h)
        assert set(train) | set(h) == set(units)


def test_kfold_rejects_bad_k() -> None:
    with pytest.raises(ValueError):
        unit_kfold([1, 2, 3], k=4)


def test_prepared_splits_are_unit_disjoint(synthetic_root: Path) -> None:
    dev = prepare_dev(
        synthetic_root / "data" / "raw" / "cmapss",
        "FD002",
        cap=125,
        window=10,
        stride=1,
        val_frac=0.2,
        split_seed=0,
        n_regimes=6,
        var_threshold=1e-4,
    )
    assert set(dev.train.units).isdisjoint(dev.val.units)
    assert dev.pre.fit_units == dev.train.units
    for data in (dev.train, dev.val):
        idx = data.index
        # Every window lies inside one unit's array: no window spans two units.
        lengths = np.array([len(a) for a in data.arrays])
        assert (idx.start >= 0).all()
        assert (idx.end < lengths[idx.segment]).all()


def test_rotation_folds_ds02_shape_and_coverage() -> None:
    from aerosentinel.data.splits import rotation_folds

    units = [2, 5, 10, 16, 18, 20]
    folds = rotation_folds(units, seed=0)
    assert len(folds) == 6
    vals, cals = [], []
    for train, val, cal in folds:
        assert len(train) == 4 and len(val) == 1 and len(cal) == 1
        assert set(train).isdisjoint(val) and set(train).isdisjoint(cal)
        assert set(val).isdisjoint(cal)
        assert sorted(train + val + cal) == units
        vals += val
        cals += cal
    assert sorted(vals) == units and sorted(cals) == units  # each held out exactly once
    assert rotation_folds(units, seed=0) == folds  # deterministic


def test_rotation_folds_rejects_too_few_units() -> None:
    import pytest

    from aerosentinel.data.splits import rotation_folds

    with pytest.raises(ValueError):
        rotation_folds([1, 2])
