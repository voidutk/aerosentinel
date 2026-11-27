"""Per-engine healthy baseline: own first 5 cycles only, never hs or other engines."""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pytest

from aerosentinel.data import baseline_residuals as br

FAST = br.BaselineConfig(steps=200, hidden=16)


def engine(seed: int, n_cycles: int = 12, rows: int = 200) -> tuple[np.ndarray, ...]:
    """Synthetic engine: X_s is a smooth function of W plus a drift that grows with cycle."""
    rng = np.random.default_rng(seed)
    cycle = np.repeat(np.arange(1, n_cycles + 1), rows)
    w = rng.normal(size=(cycle.size, 4))
    drift = 0.05 * np.maximum(cycle - 5, 0)[:, None]
    x = np.tanh(w @ rng.normal(size=(4, 14))) + drift + 0.01 * rng.normal(size=(cycle.size, 14))
    return w, x, cycle


def test_signature_has_no_health_or_label_inputs() -> None:
    params = set(inspect.signature(br.fit_engine_baseline).parameters)
    assert params == {"w", "x_s", "cycle", "cfg"}


def test_fit_sees_only_own_first_five_cycles(monkeypatch: pytest.MonkeyPatch) -> None:
    w, x, cycle = engine(0)
    seen: list[np.ndarray] = []
    orig = br._train_mlp

    def spy(wz: np.ndarray, xz: np.ndarray, cfg: br.BaselineConfig) -> Any:
        seen.append(wz)
        return orig(wz, xz, cfg)

    monkeypatch.setattr(br, "_train_mlp", spy)
    base = br.fit_engine_baseline(w, x, cycle, FAST)
    assert base.cycles == [1, 2, 3, 4, 5]
    assert base.n_fit_rows == seen[0].shape[0] == 5 * 200
    # the standardization was fit on exactly the cycle 1-5 rows
    np.testing.assert_allclose(base.w_mean, w[cycle <= 5].mean(0))


def test_later_cycles_and_other_engines_do_not_change_the_baseline() -> None:
    w, x, cycle = engine(0)
    a = br.fit_engine_baseline(w, x, cycle, FAST)
    x2 = x.copy()
    x2[cycle > 5] = 1e6  # garbage after cycle 5
    b = br.fit_engine_baseline(w, x2, cycle, FAST)
    for k in a.state:
        assert np.array_equal(a.state[k].numpy(), b.state[k].numpy())
    # another engine's data is never an input: fitting it leaves engine 0's result unchanged
    br.fit_engine_baseline(*engine(1), FAST)
    c = br.fit_engine_baseline(w, x, cycle, FAST)
    for k in a.state:
        assert np.array_equal(a.state[k].numpy(), c.state[k].numpy())


def test_residuals_near_zero_in_sample_and_drift_later() -> None:
    w, x, cycle = engine(0)
    base = br.fit_engine_baseline(w, x, cycle, br.BaselineConfig(steps=1500, hidden=32))
    r = br.residuals(base, w, x, chunk=333)
    assert r.shape == x.shape and r.dtype == np.float32
    early = np.abs(r[cycle <= 5]).mean()
    late = r[cycle == cycle.max()].mean()
    assert early < 0.3
    assert late > early  # the synthetic drift shows up as a positive residual


def test_extrapolation_share() -> None:
    w, x, cycle = engine(0)
    base = br.fit_engine_baseline(w, x, cycle, FAST)
    assert br.extrapolation_share(base, w, cycle) < 0.1
    w2 = w.copy()
    w2[cycle > 5, 0] = 99.0
    assert br.extrapolation_share(base, w2, cycle) == 1.0
    assert np.isnan(br.extrapolation_share(base, w[cycle <= 5], cycle[cycle <= 5]))
