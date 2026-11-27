"""Diagnostic: largest |z| of each sensor after per-regime normalization (dev units only).

Backs the README statement about which quantized sensors turn into spikes. For every
C-MAPSS subset it fits the Phase 1 preprocessor on train-split units (exactly as training
does) and reports, per sensor, the largest |z| and the number of rows with |z| > 10 on the
train-split and validation units. Official test files are never read.

Writes ``reports/results/cmapss-all_sensor-zmax_diag.json``.

Usage::

    uv run python scripts/sensor_spikes.py
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from hydra import compose, initialize_config_dir

from aerosentinel.data.cmapss import SENSORS, UNIT, load_train
from aerosentinel.data.splits import train_val_units
from aerosentinel.runs import CONFIG_DIR, git_info, prepare_dev_from_cfg, raw_dir
from aerosentinel.utils import get_paths

STEM = "cmapss-all_sensor-zmax_diag"
SPIKE_Z = 10.0


def _stats(z: np.ndarray) -> dict[str, Any]:
    a = np.abs(z)
    return {"max_abs_z": float(a.max()), f"n_rows_abs_z_gt_{SPIKE_Z:g}": int((a > SPIKE_Z).sum())}


def subset_report(subset: str) -> dict[str, Any]:
    """Per-sensor |z| statistics for one subset (train-split and val units)."""
    with initialize_config_dir(str(CONFIG_DIR), version_base="1.3"):
        cfg = compose("config", overrides=[f"experiment=cmapss-{subset.lower()}_ridge_cap125"])
    dev = prepare_dev_from_cfg(cfg)
    df = load_train(raw_dir(cfg), subset)
    tr, va = train_val_units(df[UNIT].unique().tolist(), float(cfg.data.val_frac), cfg.split_seed)
    z = dev.pre.normalizer.transform(df)
    sel = dev.pre.selector
    sensors: dict[str, Any] = {}
    for s in SENSORS:
        status = (
            "kept"
            if s in sel.kept_
            else next(rule for rule, names in sel.dropped_by_rule_.items() if s in names)
        )
        sensors[s] = {
            "status": status,
            "train": _stats(z.loc[z[UNIT].isin(tr), s].to_numpy()),
            "val": _stats(z.loc[z[UNIT].isin(va), s].to_numpy()),
        }
    kept_max = max(sensors[s]["train"]["max_abs_z"] for s in sel.kept_)
    return {"max_abs_z_kept_train": kept_max, "sensors": sensors}


def main() -> int:
    """Write the diagnostic JSON."""
    paths = get_paths()
    doc = {
        "experiment": STEM,
        "dataset": "cmapss-all",
        "model": "sensor-zmax",
        "setting": "diag",
        "written_by": "scripts/sensor_spikes.py",
        "note": "diagnostic, not a model result; dev units only (train-split + val), "
        "preprocessing fit on train-split units",
        "git": git_info(paths.root),
        "subsets": {s: subset_report(s) for s in ("FD001", "FD002", "FD003", "FD004")},
    }
    out = paths.results / f"{STEM}.json"
    out.write_text(json.dumps(doc, indent=2) + "\n")
    for s, r in doc["subsets"].items():
        few = {
            k: v["train"]["max_abs_z"]
            for k, v in r["sensors"].items()
            if v["status"] == "few_distinct"
        }
        print(f"{s}: few_distinct max|z| {few}; kept max|z| {r['max_abs_z_kept_train']:.1f}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
