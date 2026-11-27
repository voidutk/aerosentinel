"""README table generation from results JSONs."""

from __future__ import annotations

from typing import Any

import pytest

from synth import load_script

mt = load_script("make_tables")


def _stat(m: float, s: float = 0.5) -> dict[str, float]:
    return {"mean": m, "std": s}


def _result(
    dataset: str,
    model: str,
    rmse: float,
    alarm: bool = False,
    val: float | None = None,
    seeds: tuple[float, ...] | None = None,
) -> dict[str, Any]:
    block = {
        "rmse": _stat(rmse),
        "mae": _stat(rmse * 0.8),
        "score_sum": _stat(rmse * 30),
        "score_mean": _stat(1.0),
        "phm21": _stat(5.0),
    }
    return {
        "experiment": f"{dataset}_{model}_cap125",
        "dataset": dataset,
        "model": model,
        "setting": "cap125",
        "seeds": [0, 1, 2],
        "n_train_units": 80,
        "n_val_units": 20,
        "n_test_units": 100,
        "n_padded_test_units": 0,
        "summary": {
            "val": {**block, "rmse": _stat(rmse if val is None else val)},
            "test_capped": block,
            "test_uncapped": {**block, "rmse": _stat(rmse + 5)},
        },
        "per_seed": [
            {"seed": i, "test_capped": {"rmse": v}} for i, v in enumerate(seeds or (rmse,) * 3)
        ],
        "leakage_alarm": {"triggered": alarm},
    }


def test_table_rows_order_and_bold_best() -> None:
    res = [
        _result("cmapss-fd002", "lstm", 20.0),
        _result("cmapss-fd001", "lgbm", 12.5),
        _result("cmapss-fd001", "constant", 40.0),
        _result("ncmapss-ds02", "lgbm", 5.0),  # ignored by the C-MAPSS table
    ]
    md = mt.render_cmapss_table(res)
    rows = [ln for ln in md.splitlines() if ln.startswith("| FD")]
    assert [r.split("|")[2].strip() for r in rows] == ["Constant", "LightGBM", "LSTM"]
    assert "**12.50 ± 0.50**" in rows[1]
    assert "**20.00 ± 0.50**" in rows[2]  # best (only) row of FD002
    assert "17.50 ± 0.50" in rows[1]  # uncapped RMSE column
    assert "375 ± 0" in rows[1]  # score sum, 0 decimals
    assert "DS02" not in md
    assert "Leakage alarm" not in md


def test_alarm_is_surfaced() -> None:
    md = mt.render_cmapss_table([_result("cmapss-fd001", "lgbm", 9.0, alarm=True)])
    assert "Leakage alarm" in md


def test_targets_pick_model_on_validation_not_test() -> None:
    res = [
        # LightGBM is best on test, LSTM best on val: the LSTM must be picked.
        _result("cmapss-fd001", "lgbm", 12.5, val=15.0),
        _result("cmapss-fd001", "lstm", 14.0, val=13.0),
        _result("cmapss-fd004", "cnn1d", 21.0, val=18.0),
    ]
    md = mt.render_cmapss_targets(res)
    fd001 = next(ln for ln in md.splitlines() if ln.startswith("| FD001"))
    fd004 = next(ln for ln in md.splitlines() if ln.startswith("| FD004"))
    assert "LSTM" in fd001 and "13.00 ± 0.50" in fd001 and "14.00 ± 0.50" in fd001
    assert fd001.endswith("| **no** |")  # LSTM test RMSE 14 > 13
    assert "LightGBM" not in fd001
    assert "1D-CNN" in fd004 and "**no**" in fd004
    assert "picked on val" in md
    assert "not run" in mt.render_cmapss_targets([])


def test_target_met_on_mean_but_not_every_seed_is_inconclusive() -> None:
    res = [
        _result("cmapss-fd001", "cnn1d", 12.98, seeds=(13.01, 12.65, 13.28)),
        _result("cmapss-fd004", "cnn1d", 14.25, seeds=(14.2, 14.38, 14.17)),
    ]
    md = mt.render_cmapss_targets(res)
    fd001 = next(ln for ln in md.splitlines() if ln.startswith("| FD001"))
    fd004 = next(ln for ln in md.splitlines() if ln.startswith("| FD004"))
    assert "13.01, 12.65, 13.28" in fd001
    assert fd001.endswith("| **inconclusive** (mean meets it; 1/3 seeds do) |")
    assert fd004.endswith("| yes |")


def test_replace_between_markers_is_idempotent() -> None:
    text = "# T\n\n<!-- results:cmapss:start -->\nold\n<!-- results:cmapss:end -->\ntail\n"
    once = mt.replace_between_markers(text, "cmapss", "NEW TABLE")
    assert "old" not in once and "NEW TABLE" in once and once.endswith("tail\n")
    assert mt.replace_between_markers(once, "cmapss", "NEW TABLE") == once


def test_missing_markers_raise() -> None:
    with pytest.raises(KeyError):
        mt.replace_between_markers("no markers here", "cmapss", "x")


def test_committed_readme_matches_committed_results() -> None:
    from aerosentinel.utils import get_paths

    paths = get_paths()
    readme = (paths.root / "README.md").read_text()
    assert mt.render_readme(readme, mt.load_results(paths.results)) == readme, (
        "README tables are stale: run `uv run python scripts/make_tables.py`"
    )


def _ncm(setting: str, res: bool, rmse: float) -> dict[str, Any]:
    block = {"rmse": _stat(rmse), "mae": _stat(rmse), "score_mean": _stat(0.4)}
    stem = f"single-{setting}-{'res' if res else 'nores'}-cap48-ds10"
    return {
        "experiment": f"ncmapss-ds02_lgbm_{stem}",
        "dataset": "ncmapss-ds02",
        "model": "lgbm",
        "setting": stem,
        "seeds": [0, 1, 2],
        "protocol": {"inputs": ["W", "X_s", "residuals"] if res else ["W", "X_s"]},
        "config": {"data": {"cap": 48, "window": 64, "downsample_factor": 10, "stride": 10}},
        "test_units": [11, 14, 15],
        "excluded_test_cycles": {"total": {"n_scored": 190, "n_without_window": 7}},
        "summary": {
            "oof_val_per_cycle_capped": block,
            "test_per_cycle_capped": block,
            "test_per_cycle_uncapped": {**block, "rmse": _stat(rmse + 9)},
            "test_per_cycle_ge6_capped": block,
            "test_per_window_capped": {**block, "rmse": _stat(rmse + 1)},
            "test_per_unit_per_cycle_capped": {"11": block, "14": None, "15": block},
        },
    }


def test_ncmapss_table_orders_rows_and_bolds_best() -> None:
    res = [
        _ncm("cruise", False, 6.0),
        _ncm("full", True, 4.0),
        _ncm("cruise", True, 5.0),
        _ncm("full", False, 7.0),
        _result("cmapss-fd001", "lgbm", 12.5),  # ignored
    ]
    md = mt.render_ncmapss_table(res)
    rows = [ln for ln in md.splitlines() if ln.startswith("| DS02")]
    assert [(r.split("|")[3].strip(), r.split("|")[4].strip()) for r in rows] == [
        ("full flight", "W + X_s + residuals"),
        ("full flight", "W + X_s"),
        ("cruise only", "W + X_s + residuals"),
        ("cruise only", "W + X_s"),
    ]
    assert "**4.00 ± 0.50**" in rows[0] and "**" not in rows[1].split("|")[6]
    assert "13.00 ± 0.50" in rows[0]  # uncapped column
    assert "190 / 7" in rows[0]
    per_unit = [ln for ln in md.splitlines() if ln.startswith("| full flight |")]
    assert "–" in per_unit[0]  # unit 14 without scored cycles
    assert "T = 64" in md and "cycles ≥ 6" in md
    assert "No N-CMAPSS results" in mt.render_ncmapss_table([_result("cmapss-fd001", "x", 1)])


def test_ncmapss_data_block_renders_audits() -> None:
    fc = {
        "experiment": "ncmapss-ds02_data_fc-audit",
        "dataset": "ncmapss-ds02",
        "model": "data",
        "audit": {"DS02": {"dev_units_per_class": {"1": [], "2": [], "3": [2, 5]}}},
        "lofco_dataset": "DS03",
    }
    cap = {
        "experiment": "ncmapss-pooled_data_cap-check",
        "dataset": "ncmapss-pooled",
        "model": "data",
        "cruisebench": {
            "reachable": True,
            "commit": "ef26d40fe06b75ce",
            "caps_by_eta": {"eta5": {"caps": {"DS02": 34}}, "eta0": {"caps": {"DS02": 41}}},
        },
        "cap_source": "hs-derived",
        "cap_source_reason": "definition not stated",
        "spec_caps_eta5": {"DS02": 34},
        "hs_derived": {"caps": {"DS02": 48}},
        "caps_used": {"DS02": 48},
    }
    md = mt.render_ncmapss_data([fc, cap])
    assert "**DS03**" in md and "class 3: 2" in md
    assert "| DS02 | 34 | 34 | 41 | 48 | 48 |" in md
    assert "hs-derived" in md and "ef26d40fe0" in md
    assert "No N-CMAPSS data audits" in mt.render_ncmapss_data([])
