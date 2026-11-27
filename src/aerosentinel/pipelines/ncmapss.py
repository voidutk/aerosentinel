"""N-CMAPSS LightGBM pipeline: unit-rotation training and final test evaluation.

Protocol (SPEC §1.3):

- Inputs are W + X_s, plus the healthy-baseline residuals when ``data.residuals`` is true.
  Residual models train and are scored on cycles >= 6 (cycles 1-5 are in-sample for the
  baseline). Every result also reports a "cycles >= 6" block so runs compare like for like.
- Splits are by unit. The dev units rotate through ``rotation_folds``: per fold 4 train /
  1 val (early stopping) / 1 calib unit. The calib unit's predictions are out-of-fold, and
  every dev unit is a calib unit exactly once. Official test units are read only by
  :func:`evaluate_ncmapss`; the test prediction is the mean of the fold models.
- One ``data.stride`` serves every split. Windows never cross a (unit, cycle) boundary or a
  cruise-run edge. Per-cycle prediction = median of the cycle's window predictions; per-cycle
  metrics are primary, per-window secondary. Targets are capped at ``data.cap``; truth is
  reported capped (primary) and uncapped.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from omegaconf import DictConfig

from aerosentinel.data.masks import CruiseMaskParams
from aerosentinel.data.ncmapss import UnitStore
from aerosentinel.data.ncmapss_windows import channel_names, iter_batches, window_index
from aerosentinel.data.splits import rotation_folds
from aerosentinel.data.windows import aggregate_per_cycle
from aerosentinel.metrics import point_metrics
from aerosentinel.models.lgbm import feature_names, train_booster, window_features
from aerosentinel.runs import cfg_dict, config_hash, git_info, run_dir
from aerosentinel.utils import get_paths

#: Cycles 1..BASELINE_CYCLES are in-sample for the residual baseline.
FIRST_SCORED_CYCLE = 6
F32 = NDArray[np.float32]


def store_root() -> Path:
    """``data/processed/ncmapss`` of the current project root."""
    return get_paths().data_processed / "ncmapss"


def channel_groups(cfg: DictConfig) -> list[str]:
    """Input channel groups of the run: W, X_s (+ residuals)."""
    return ["W", "X_s", "residuals"] if bool(cfg.data.residuals) else ["W", "X_s"]


def min_cycle(cfg: DictConfig) -> int:
    """First cycle a model trains on and is scored on."""
    return FIRST_SCORED_CYCLE if bool(cfg.data.residuals) else 1


def mask_params(cfg: DictConfig) -> CruiseMaskParams:
    """Cruise-mask parameters from the data config."""
    return CruiseMaskParams(**dict(cfg.data.mask))


@dataclass
class Features:
    """Engineered window features with their keys and raw (uncapped) end labels."""

    x: F32
    y: F32
    unit: NDArray[np.int64]
    cycle: NDArray[np.int64]
    names: list[str]


def build_features(cfg: DictConfig, store: UnitStore, units: list[int]) -> Features:
    """Window index over ``units`` -> LightGBM window features, gathered batch by batch."""
    d = cfg.data
    groups = channel_groups(cfg)
    idx = window_index(
        store,
        units,
        str(d.setting),
        mask_params(cfg),
        window=int(d.window),
        stride=int(d.stride),
        downsample=int(d.downsample_factor),
        min_cycle=min_cycle(cfg),
        cache_dir=store_root() / str(d.subset) / "index",
    )
    xs, ys, us, cs = [], [], [], []
    for b in iter_batches(store, idx, groups, batch_size=2048):
        xs.append(window_features(b.x))
        ys.append(b.y)
        us.append(b.unit)
        cs.append(b.cycle)
    names = feature_names(channel_names(store, groups), regime=False)
    if not xs:
        e = np.zeros(0, dtype=np.int64)
        return Features(np.zeros((0, len(names)), np.float32), np.zeros(0, np.float32), e, e, names)
    return Features(
        np.concatenate(xs), np.concatenate(ys), np.concatenate(us), np.concatenate(cs), names
    )


def cap_labels(y: NDArray[Any], cap: int | None) -> F32:
    """Piecewise-linear cap: ``min(y, cap)`` (no-op for ``cap=None``)."""
    yy = np.asarray(y, dtype=np.float32)
    return yy if cap is None else np.minimum(yy, np.float32(cap))


def per_cycle_frame(f: Features, pred: NDArray[Any], cap: int | None) -> pd.DataFrame:
    """Per-cycle prediction (median over the cycle's windows) and truth."""
    u, c, p = aggregate_per_cycle(np.asarray(pred, np.float64), f.unit, f.cycle)
    _, _, y = aggregate_per_cycle(f.y.astype(np.float64), f.unit, f.cycle)
    _, counts = np.unique(np.stack([f.unit, f.cycle], axis=1), axis=0, return_counts=True)
    return pd.DataFrame(
        {
            "unit": u,
            "cycle": c,
            "n_windows": counts,
            "rul_true": y,
            "rul_true_capped": cap_labels(y, cap).astype(np.float64),
            "pred": p,
        }
    )


def train_ncmapss(cfg: DictConfig) -> Path:
    """Train one seed: one LightGBM model per rotation fold; save boosters + OOF predictions.

    Never reads official test units.

    Returns
    -------
    pathlib.Path
        The run directory.
    """
    seed = int(cfg.seed)
    cap = None if cfg.data.cap is None else int(cfg.data.cap)
    store = UnitStore(store_root(), str(cfg.data.subset), "dev")
    t0 = time.perf_counter()
    feats = build_features(cfg, store, store.units)
    y_cap = cap_labels(feats.y, cap)
    folds = rotation_folds(store.units, seed=int(cfg.split_seed))
    rd = run_dir(get_paths(), str(cfg.name), seed)
    rd.mkdir(parents=True, exist_ok=True)
    oof = np.full(len(feats.y), np.nan, dtype=np.float64)
    fold_info = []
    n_leaves = 0
    for i, (tr, va, ca) in enumerate(folds):
        mtr, mva, mca = (np.isin(feats.unit, g) for g in (tr, va, ca))
        booster = train_booster(
            feats.x[mtr],
            y_cap[mtr],
            feats.x[mva],
            y_cap[mva],
            feats.names,
            dict(cfg.model.params),
            seed,
            int(cfg.model.early_stopping_rounds),
        )
        booster.save_model(str(rd / f"fold{i}.txt"), num_iteration=booster.best_iteration)
        oof[mca] = booster.predict(feats.x[mca], num_iteration=booster.best_iteration)
        dump = booster.dump_model(num_iteration=booster.best_iteration)
        n_leaves += int(sum(t["num_leaves"] for t in dump["tree_info"]))
        fold_info.append(
            {
                "fold": i,
                "train_units": tr,
                "val_units": va,
                "calib_units": ca,
                "n_train_windows": int(mtr.sum()),
                "best_iteration": int(booster.best_iteration),
            }
        )
    if np.isnan(oof).any():
        raise RuntimeError("some dev windows received no out-of-fold prediction")
    cyc = per_cycle_frame(feats, oof, cap)
    cyc.to_csv(rd / "oof_per_cycle.csv", index=False)
    meta: dict[str, Any] = {
        "experiment": str(cfg.name),
        "seed": seed,
        "config_hash": config_hash(cfg),
        "config": cfg_dict(cfg),
        "dev_units": store.units,
        "folds": fold_info,
        "input_names": feats.names,
        "channels": channel_names(store, channel_groups(cfg)),
        "n_dev_windows": len(feats.y),
        "oof_per_cycle_capped": point_metrics(cyc["rul_true_capped"], cyc["pred"]),
        "oof_per_window_capped": point_metrics(y_cap, oof),
        "n_oof_cycles": len(cyc),
        "n_params": n_leaves,
        "train_seconds": time.perf_counter() - t0,
        "git": git_info(get_paths().root),
    }
    (rd / "train_meta.json").write_text(json.dumps(meta, indent=2))
    r = meta["oof_per_cycle_capped"]["rmse"]
    print(f"[train] {cfg.name} seed={seed} OOF per-cycle RMSE (capped) {r:.3f} -> {rd}")
    return rd


# --- evaluation ----------------------------------------------------------------------------


class StaleRunError(RuntimeError):
    """A saved run was trained with a different config than the one being evaluated."""


def _summary(per_seed: list[dict[str, Any]], path: list[str]) -> dict[str, Any] | None:
    """Mean and sample std (ddof=1) of the metric dict found at ``path`` in each seed."""
    vals = []
    for s in per_seed:
        d: Any = s
        for k in path:
            d = d[k]
        vals.append(d)
    if any(v is None for v in vals):
        return None
    out = {}
    for m in vals[0]:
        v = np.array([x[m] for x in vals], dtype=np.float64)
        out[m] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0}
    return out


def _metric_block(df: pd.DataFrame) -> dict[str, Any] | None:
    if df.empty:
        return None
    return {
        "n_cycles": len(df),
        "capped": point_metrics(df["rul_true_capped"], df["pred"]),
        "uncapped": point_metrics(df["rul_true"], df["pred"]),
    }


def _excluded_cycles(store: UnitStore, cyc: pd.DataFrame, first: int) -> dict[str, Any]:
    """Cycles of the evaluated units without a valid window, per unit and in total."""
    per_unit: dict[str, Any] = {}
    for u in store.units:
        fl = np.load(store.unit_dir(u) / "flights.npy")
        eligible = set(int(c) for c in fl[:, 0] if c >= first)
        scored = set(cyc.loc[cyc["unit"] == u, "cycle"].astype(int))
        per_unit[str(u)] = {
            "n_cycles": int(len(fl)),
            "n_before_first_scored_cycle": int((fl[:, 0] < first).sum()),
            "n_without_window": len(eligible - scored),
            "n_scored": len(scored),
        }
    tot = {k: sum(v[k] for v in per_unit.values()) for k in next(iter(per_unit.values()))}
    return {"total": tot, "per_unit": per_unit}


def evaluate_ncmapss(cfg: DictConfig) -> dict[str, Any]:
    """Score every seed on the official test units and write ``reports/results/<name>.json``.

    The only N-CMAPSS code path that opens the test split (``allow_test=True``).
    """
    paths = get_paths()
    name = str(cfg.name)
    cap = None if cfg.data.cap is None else int(cfg.data.cap)
    expected = config_hash(cfg)
    test = UnitStore(store_root(), str(cfg.data.subset), "test", allow_test=True)
    feats = build_features(cfg, test, test.units)
    first = min_cycle(cfg)
    per_seed: list[dict[str, Any]] = []
    meta: dict[str, Any] = {}
    excluded: dict[str, Any] = {}
    for seed in [int(s) for s in cfg.seeds]:
        rd = run_dir(paths, name, seed)
        meta = json.loads((rd / "train_meta.json").read_text())
        if meta["config_hash"] != expected:
            raise StaleRunError(f"{rd}: trained with {meta['config_hash']}, evaluating {expected}")
        boosters = [lgb.Booster(model_file=str(rd / f"fold{f['fold']}.txt")) for f in meta["folds"]]
        pred = np.mean([b.predict(feats.x) for b in boosters], axis=0)
        cyc = per_cycle_frame(feats, pred, cap)
        cyc.to_csv(rd / "test_per_cycle.csv", index=False)
        excluded = _excluded_cycles(test, cyc, first)
        late = cyc[cyc["cycle"] >= FIRST_SCORED_CYCLE]
        per_seed.append(
            {
                "seed": seed,
                "oof_val": {
                    "per_cycle_capped": meta["oof_per_cycle_capped"],
                    "per_window_capped": meta["oof_per_window_capped"],
                },
                "test_per_cycle": _metric_block(cyc),
                "test_per_cycle_cycles_ge6": _metric_block(late),
                "test_per_window": {
                    "capped": point_metrics(cap_labels(feats.y, cap), pred),
                    "uncapped": point_metrics(feats.y, pred),
                },
                "test_per_unit_per_cycle": {
                    str(u): _metric_block(cyc[cyc["unit"] == u]) for u in test.units
                },
                "best_iterations": [f["best_iteration"] for f in meta["folds"]],
                "n_params": meta["n_params"],
                "train_seconds": meta["train_seconds"],
                "train_git": meta["git"],
            }
        )
    dataset, model_name, setting = name.split("_", 2)
    summary: dict[str, Any] = {
        "oof_val_per_cycle_capped": _summary(per_seed, ["oof_val", "per_cycle_capped"]),
        "test_per_cycle_capped": _summary(per_seed, ["test_per_cycle", "capped"]),
        "test_per_cycle_uncapped": _summary(per_seed, ["test_per_cycle", "uncapped"]),
        "test_per_cycle_ge6_capped": _summary(per_seed, ["test_per_cycle_cycles_ge6", "capped"]),
        "test_per_window_capped": _summary(per_seed, ["test_per_window", "capped"]),
        "test_per_window_uncapped": _summary(per_seed, ["test_per_window", "uncapped"]),
        "test_per_unit_per_cycle_capped": {
            str(u): _summary(per_seed, ["test_per_unit_per_cycle", str(u), "capped"])
            for u in test.units
        },
    }
    doc: dict[str, Any] = {
        "experiment": name,
        "dataset": dataset,
        "model": model_name,
        "setting": setting,
        "written_by": "aerosentinel.evaluate (pipelines.ncmapss)",
        "protocol": {
            "inputs": channel_groups(cfg),
            "first_scored_cycle": first,
            "split": "unit-level rotation over dev units: per fold 4 train / 1 val (early "
            "stopping) / 1 calib (out-of-fold prediction); every dev unit is calib once",
            "oof_val": "per-cycle metrics on the calib unit of each fold (out-of-fold)",
            "test": "official test units; prediction = mean of the fold models' window "
            "predictions; per-cycle = median of the cycle's window predictions",
            "setting": str(cfg.data.setting),
            "windows": f"T={cfg.data.window} model samples, stride={cfg.data.stride} model "
            f"samples (same for every split), downsample={cfg.data.downsample_factor}; never "
            "crossing a (unit, cycle) boundary or a cruise-run edge",
            "truth": f"capped at {cap} (primary) and uncapped (secondary)",
            "score": "Saxena score with d = pred - true; score_mean is the N-CMAPSS convention",
            "std": "sample std over seeds (ddof=1)",
        },
        "config_hash": expected,
        "config": cfg_dict(cfg),
        "git": git_info(paths.root),
        "seeds": [int(s) for s in cfg.seeds],
        "dev_units": meta.get("dev_units"),
        "test_units": test.units,
        "folds": meta.get("folds"),
        "channels": meta.get("channels"),
        "n_inputs": len(meta.get("input_names", [])),
        "n_dev_windows": meta.get("n_dev_windows"),
        "n_test_windows": len(feats.y),
        "excluded_test_cycles": excluded,
        "summary": summary,
        "per_seed": per_seed,
    }
    doc["config"].pop("seed", None)
    paths.results.mkdir(parents=True, exist_ok=True)
    out = paths.results / f"{name}.json"
    out.write_text(json.dumps(doc, indent=2) + "\n")
    r = summary["test_per_cycle_capped"]["rmse"]
    print(f"[evaluate] {name}: per-cycle RMSE (capped) {r['mean']:.2f} ± {r['std']:.2f} -> {out}")
    return doc
