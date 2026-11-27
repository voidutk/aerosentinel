"""No banned column reaches any model, and preprocessing is fit on train-split units only."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from aerosentinel.data import cmapss as cm
from aerosentinel.data.prepare import prepare_dev
from aerosentinel.models import MODEL_NAMES, build_model
from aerosentinel.models.base import ALLOWED_INPUT_BASES, BANNED_INPUTS, feature_base

MODEL_CFGS: dict[str, dict[str, Any]] = {
    "constant": {"name": "constant"},
    "ridge": {"name": "ridge", "alphas": [1.0]},
    "lgbm": {"name": "lgbm", "early_stopping_rounds": 5, "params": {"n_estimators": 20}},
    "cnn1d": {
        "name": "cnn1d",
        "arch": {"n_conv": 1, "filters": 2, "kernel": 3, "fc": 4},
        "train": {"max_epochs": 1, "batch_size": 64},
    },
    "lstm": {
        "name": "lstm",
        "arch": {"hidden": 4, "layers": 2},
        "train": {"max_epochs": 1, "batch_size": 64},
    },
}


def _dev(root: Path, subset: str = "FD002") -> Any:
    return prepare_dev(
        root / "data" / "raw" / "cmapss",
        subset,
        cap=125,
        window=10,
        stride=1,
        val_frac=0.2,
        split_seed=0,
        n_regimes=6 if subset == "FD002" else 1,
        var_threshold=1e-4,
    )


def test_banned_and_allowed_sets_are_disjoint() -> None:
    assert ALLOWED_INPUT_BASES.isdisjoint(BANNED_INPUTS)
    assert set(MODEL_CFGS) == set(MODEL_NAMES)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_model_inputs_only_sensors_or_regime(synthetic_root: Path, name: str) -> None:
    dev = _dev(synthetic_root)
    model = build_model(MODEL_CFGS[name], target_scale=125.0)
    model.fit(dev.train, dev.val, seed=0)
    bases = {feature_base(n) for n in model.input_names()}
    assert bases <= ALLOWED_INPUT_BASES, bases - ALLOWED_INPUT_BASES
    assert not bases & BANNED_INPUTS
    if name in ("cnn1d", "lstm", "ridge"):
        assert "regime_id" not in bases  # regime info enters only LightGBM (SPEC §5.1)


def test_channels_exclude_settings_unit_cycle_rul(synthetic_root: Path) -> None:
    dev = _dev(synthetic_root)
    assert set(dev.pre.channels) <= set(cm.SENSORS)
    # s1, s5 are constant within each synthetic regime -> dropped after normalization.
    assert {"s1", "s5"} <= set(dev.pre.selector.dropped_)


def test_preprocessing_fit_sees_only_train_split_units(
    synthetic_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, set[int]] = {}
    orig_norm, orig_sel = cm.RegimeNormalizer.fit, cm.SensorSelector.fit

    def norm_fit(self: cm.RegimeNormalizer, df: pd.DataFrame) -> cm.RegimeNormalizer:
        seen["normalizer"] = set(df["unit"].unique().tolist())
        return orig_norm(self, df)

    def sel_fit(self: cm.SensorSelector, df: pd.DataFrame) -> cm.SensorSelector:
        seen["selector"] = set(df["unit"].unique().tolist())
        return orig_sel(self, df)

    monkeypatch.setattr(cm.RegimeNormalizer, "fit", norm_fit)
    monkeypatch.setattr(cm.SensorSelector, "fit", sel_fit)
    dev = _dev(synthetic_root)
    assert seen["normalizer"] == set(dev.train.units)
    assert seen["selector"] == set(dev.train.units)
    assert seen["normalizer"].isdisjoint(dev.val.units)


def test_prepare_dev_never_reads_test_files(
    synthetic_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = synthetic_root / "data" / "raw" / "cmapss"
    for f in raw.glob("test_*.txt"):
        f.unlink()
    for f in raw.glob("RUL_*.txt"):
        f.unlink()
    dev = _dev(synthetic_root)  # must succeed without any test file
    assert len(dev.train.index) > 0
