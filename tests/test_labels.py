"""RUL labels: piecewise cap and monotonicity within a unit."""

from __future__ import annotations

import numpy as np
import pandas as pd

from aerosentinel.data.cmapss import add_rul, cap_truth
from synth import synthetic_cmapss


def test_piecewise_cap() -> None:
    df = pd.DataFrame({"unit": [1] * 200, "cycle": np.arange(1, 201)})
    out = add_rul(df, cap=125)
    rul = out["RUL"].to_numpy()
    assert rul.max() == 125
    assert rul[-1] == 0
    # Linear part: last 126 cycles count down 125..0; before that, flat at the cap.
    np.testing.assert_array_equal(rul[-126:], np.arange(125, -1, -1))
    assert (rul[:-126] == 125).all()


def test_uncapped_when_cap_none() -> None:
    df = pd.DataFrame({"unit": [1] * 200, "cycle": np.arange(1, 201)})
    rul = add_rul(df, cap=None)["RUL"].to_numpy()
    np.testing.assert_array_equal(rul, np.arange(199, -1, -1))


def test_rul_non_increasing_within_unit() -> None:
    df, _ = synthetic_cmapss(10, n_regimes=6, seed=0)
    out = add_rul(df.sample(frac=1.0, random_state=0), cap=125)  # shuffled rows
    for _, g in out.sort_values(["unit", "cycle"]).groupby("unit"):
        assert (np.diff(g["RUL"].to_numpy()) <= 0).all()
        assert g["RUL"].iloc[-1] == 0


def test_cap_truth() -> None:
    np.testing.assert_array_equal(cap_truth(np.array([10, 125, 200]), 125), [10, 125, 125])
    np.testing.assert_array_equal(cap_truth(np.array([10, 200]), None), [10, 200])
