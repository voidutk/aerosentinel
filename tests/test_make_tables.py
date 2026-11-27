"""README table generation from results JSONs."""

from __future__ import annotations

from typing import Any

import pytest

from synth import load_script

mt = load_script("make_tables")


def _stat(m: float, s: float = 0.5) -> dict[str, float]:
    return {"mean": m, "std": s}


def _result(
    dataset: str, model: str, rmse: float, alarm: bool = False, val: float | None = None
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
