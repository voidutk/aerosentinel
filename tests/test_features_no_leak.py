"""No banned column reaches any model, and preprocessing is fit on train-split units only."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from aerosentinel.data import cmapss as cm
from aerosentinel.data.prepare import prepare_dev
from aerosentinel.data.splits import train_val_units
from aerosentinel.models import MODEL_NAMES, build_model
from aerosentinel.models.base import ALLOWED_INPUT_BASES, BANNED_INPUTS, feature_base
from synth import write_cmapss_txt

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


def _dev(root: Path, subset: str = "FD002", cap: int | None = 125) -> Any:
    return prepare_dev(
        root / "data" / "raw" / "cmapss",
        subset,
        cap=cap,
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

    def sel_fit(self: cm.SensorSelector, df: pd.DataFrame, raw: pd.DataFrame) -> cm.SensorSelector:
        seen["selector"] = set(df["unit"].unique().tolist())
        seen["selector_raw"] = set(raw["unit"].unique().tolist())
        return orig_sel(self, df, raw)

    monkeypatch.setattr(cm.RegimeNormalizer, "fit", norm_fit)
    monkeypatch.setattr(cm.SensorSelector, "fit", sel_fit)
    dev = _dev(synthetic_root)
    assert seen["normalizer"] == set(dev.train.units)
    assert seen["selector"] == set(dev.train.units)
    assert seen["selector_raw"] == set(dev.train.units)
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


@pytest.mark.parametrize("subset", ["FD001", "FD002"])
def test_model_arrays_are_hand_zscored_kept_sensors(synthetic_root: Path, subset: str) -> None:
    """The arrays models receive equal an independent per-regime z-score of the kept sensors.

    Only the regime *assignment* is reused from the fitted normalizer; means and stds are
    recomputed here from the raw train-split rows.
    """
    dev = _dev(synthetic_root, subset)
    raw = cm.load_train(synthetic_root / "data" / "raw" / "cmapss", subset)
    tr_units, _ = train_val_units(raw["unit"].unique().tolist(), 0.2, 0)
    raw = raw.assign(regime=dev.pre.normalizer.regimes(raw))
    ch = dev.pre.channels
    fit = raw[raw["unit"].isin(tr_units)]
    mu = fit.groupby("regime")[ch].mean()
    sd = fit.groupby("regime")[ch].std(ddof=0)
    for wd in (dev.train, dev.val):
        for u, arr in zip(wd.units, wd.arrays, strict=True):
            g = raw[raw["unit"] == u].sort_values("cycle")
            r = g["regime"].to_numpy()
            want = (g[ch].to_numpy() - mu.loc[r].to_numpy()) / sd.loc[r].to_numpy()
            assert arr.shape == want.shape
            np.testing.assert_allclose(arr, want, rtol=1e-4, atol=1e-4)


def test_model_arrays_ignore_unit_cycle_settings_and_labels(synthetic_root: Path) -> None:
    """Changing unit ids, cycle numbers, settings or the RUL labels leaves X unchanged."""
    raw_dir = synthetic_root / "data" / "raw" / "cmapss"
    base = _dev(synthetic_root, "FD002")
    uncapped = _dev(synthetic_root, "FD002", cap=20)  # synthetic lives < 125: cap lower
    assert not all(
        np.array_equal(a, b) for a, b in zip(base.train.labels, uncapped.train.labels, strict=True)
    )
    df = cm.load_train(raw_dir, "FD002")
    df["unit"] += 1000  # monotonic relabel keeps the positional unit split
    df["cycle"] += 500
    # settings noise far below the regime spacing: same regimes, different settings values
    df[list(cm.SETTINGS)] += np.random.default_rng(0).normal(0, 1e-3, (len(df), 3))
    write_cmapss_txt(df, raw_dir / "train_FD002.txt")
    moved = _dev(synthetic_root, "FD002")
    for other in (uncapped, moved):
        assert other.pre.channels == base.pre.channels
        for wa, wb in ((base.train, other.train), (base.val, other.val)):
            assert len(wa.arrays) == len(wb.arrays)
            for a, b in zip(wa.arrays, wb.arrays, strict=True):
                np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4)


def test_selector_drops_quantized_sensors_within_every_regime() -> None:
    rng = np.random.default_rng(0)
    n = 400
    regime = np.repeat([0, 1], n // 2)
    raw = pd.DataFrame({s: rng.normal(0, 1, n) for s in cm.SENSORS})
    raw["s1"] = 3.0  # constant -> variance rule
    raw["s6"] = np.where(rng.random(n) < 0.98, 21.61, 21.60)  # 2 levels everywhere -> drop
    # 2 levels per regime but 4 overall -> still "<= 2 in every regime" -> drop
    raw["s10"] = np.where(regime == 0, 1.30, 1.02) + 0.01 * (rng.random(n) < 0.05)
    # 3 levels in regime 1 -> kept
    raw["s16"] = np.where(regime == 0, 0.02, 0.03 + 0.01 * rng.integers(0, 3, n))
    sd = raw.groupby(regime).transform("std")
    norm = (raw - raw.groupby(regime).transform("mean")) / sd.where(sd > 1e-9, np.inf)
    norm = norm.assign(regime=regime)
    sel = cm.SensorSelector(threshold=1e-4, max_distinct=2).fit(norm, raw)
    assert sel.dropped_by_rule_ == {"variance": ["s1"], "few_distinct": ["s6", "s10"]}
    assert sel.dropped_ == ["s1", "s6", "s10"]
    assert "s16" in sel.kept_
    off = cm.SensorSelector(threshold=1e-4, max_distinct=0).fit(norm, raw)
    assert off.dropped_ == ["s1"]
    with pytest.raises(ValueError, match="same rows"):
        sel.fit(norm, raw.iloc[1:])
