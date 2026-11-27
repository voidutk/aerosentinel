"""N-CMAPSS LightGBM pipeline end to end on a synthetic store, plus input-leak checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from aerosentinel.data import ncmapss as nc
from aerosentinel.data.baseline_residuals import BaselineConfig
from aerosentinel.data.masks import CruiseMaskParams
from aerosentinel.data.ncmapss_derive import compute_cruise_runs, compute_residuals
from aerosentinel.models.base import feature_base
from aerosentinel.pipelines import ncmapss as pipe
from synth_ncmapss import A_VAR, T_VAR, W_VAR, XS_VAR, XV_VAR, write_synthetic_ncmapss

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
DEV, TEST = [2, 5, 10, 16, 18, 20], [11, 14]
ALLOWED = set(W_VAR) | set(XS_VAR) | {f"r_{s}" for s in XS_VAR}
#: Never a model input: virtual sensors, health parameters, metadata, labels.
BANNED = set(XV_VAR) | set(T_VAR) | set(A_VAR) | {"Y", "RUL", "rul"}
SMALL_MASK = {"min_len": 128, "std_max": 50.0, "med_win": 31, "stab_win": 61}


@pytest.fixture(scope="module")
def ncmapss_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("proj")
    h5 = write_synthetic_ncmapss(root / "raw" / "N-CMAPSS_DS02-006.h5", DEV, TEST, rows=600)
    for split in ("dev", "test"):
        nc.convert_split(h5, split, root / "data" / "processed" / "ncmapss" / "DS02" / split)
        store = nc.UnitStore(
            root / "data" / "processed" / "ncmapss", "DS02", split, allow_test=True
        )
        compute_cruise_runs(store, CruiseMaskParams(**SMALL_MASK))
        compute_residuals(store, BaselineConfig(steps=100, hidden=8))
    return root


def _cfg(setting: str, residuals: bool) -> DictConfig:
    stem = f"ncmapss-ds02_lgbm_single-{'full' if setting == 'full_flight' else 'cruise'}-" + (
        "res-cap48-ds10" if residuals else "nores-cap48-ds10"
    )
    with initialize_config_dir(str(CONFIGS), version_base="1.3"):
        return compose(
            "config",
            overrides=[
                f"experiment={stem}",
                "data.window=8",
                "data.stride=4",
                "data.downsample_factor=2",
                *(f"data.mask.{k}={v}" for k, v in SMALL_MASK.items()),
                "model.early_stopping_rounds=5",
                "model.params.n_estimators=30",
                "seeds=[0,1]",
            ],
        )


@pytest.mark.parametrize(("setting", "residuals"), [("full_flight", True), ("cruise_only", False)])
def test_train_then_evaluate(
    ncmapss_root: Path, monkeypatch: pytest.MonkeyPatch, setting: str, residuals: bool
) -> None:
    monkeypatch.setenv("AEROSENTINEL_ROOT", str(ncmapss_root))
    cfg = _cfg(setting, residuals)
    opened: list[str] = []
    orig_init = nc.UnitStore.__init__

    def spy(self: nc.UnitStore, root: Path, ds: str, split: str, **kw: Any) -> None:
        opened.append(split)
        orig_init(self, root, ds, split, **kw)

    monkeypatch.setattr(nc.UnitStore, "__init__", spy)
    for seed in (0, 1):
        cfg.seed = seed
        rd = pipe.train_ncmapss(cfg)
    assert "test" not in opened  # training never opens the test split
    meta = json.loads((rd / "train_meta.json").read_text())
    # every dev unit is predicted out of fold exactly once, as a calibration unit
    calib = sorted(u for f in meta["folds"] for u in f["calib_units"])
    assert calib == DEV
    for f in meta["folds"]:
        assert not set(f["calib_units"]) & (set(f["train_units"]) | set(f["val_units"]))
    # input names: only W, X_s and residual-derived features
    bases = {feature_base(n) for n in meta["input_names"]}
    assert bases <= ALLOWED and not bases & BANNED
    assert any(b.startswith("r_") for b in bases) == residuals

    doc = pipe.evaluate_ncmapss(cfg)
    assert opened[-1] == "test"
    out = ncmapss_root / "reports" / "results" / f"{cfg.name}.json"
    assert out.is_file()
    s = doc["summary"]
    assert s["test_per_cycle_capped"]["rmse"]["mean"] > 0
    assert set(s["test_per_unit_per_cycle_capped"]) == {str(u) for u in TEST}
    cyc = np.loadtxt(rd / "test_per_cycle.csv", delimiter=",", skiprows=1)
    assert cyc[:, 1].min() >= (6 if residuals else 1)  # residual runs score from cycle 6
    assert np.all(cyc[:, 4] <= 48)  # capped truth
    ex = doc["excluded_test_cycles"]["total"]
    assert (
        ex["n_scored"] + ex["n_without_window"] + ex["n_before_first_scored_cycle"]
        == (ex["n_cycles"])
    )
    if setting == "cruise_only":
        assert ex["n_without_window"] > 0  # short cycle-2 flights have no cruise run


def test_stale_run_is_refused(ncmapss_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AEROSENTINEL_ROOT", str(ncmapss_root))
    cfg = _cfg("full_flight", True)
    cfg.seed = 0
    pipe.train_ncmapss(cfg)
    cfg.seeds = [0]
    cfg.data.stride = 3  # evaluating a different config than the one trained
    with pytest.raises(pipe.StaleRunError):
        pipe.evaluate_ncmapss(cfg)


def test_cap_labels() -> None:
    np.testing.assert_array_equal(pipe.cap_labels(np.array([60, 48, 3]), 48), [48, 48, 3])
    np.testing.assert_array_equal(pipe.cap_labels(np.array([60]), None), [60])
