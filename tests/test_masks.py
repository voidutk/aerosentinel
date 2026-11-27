"""Cruise-stage mask on synthetic altitude profiles (SPEC §1.3 item 4)."""

from __future__ import annotations

import numpy as np
import pytest

from aerosentinel.data.masks import (
    CruiseMaskParams,
    cruise_run,
    interval_iou,
    run_mask,
    true_runs,
)


def profile(segments: list[tuple[int, float]], noise: float = 5.0, seed: int = 0) -> np.ndarray:
    """Piecewise-linear altitude: each (n, target) ramps from the previous level to target."""
    rng = np.random.default_rng(seed)
    level, parts = 0.0, []
    for n, target in segments:
        parts.append(np.linspace(level, target, n, endpoint=False))
        level = target
    alt = np.concatenate(parts)
    return alt + rng.normal(0.0, noise, alt.size)


def test_true_runs() -> None:
    assert true_runs([0, 1, 1, 0, 1]) == [(1, 3), (4, 5)]
    assert true_runs([]) == []
    assert true_runs([1, 1]) == [(0, 2)]


def test_simple_cruise_is_found_inside_plateau() -> None:
    # climb 600, cruise 2000 @ 35k, descent 600
    alt = profile([(600, 35000), (2000, 35000), (600, 0)])
    run = cruise_run(alt)
    assert run is not None
    s, e = run
    assert 600 <= s < 650 and 2550 < e <= 2600  # inside the plateau, near its edges
    assert alt[s:e].std() < 50.0


def test_step_climb_picks_longest_plateau_and_never_splices() -> None:
    # climb, 900 @ 31k, step climb, 1500 @ 35k, descent
    alt = profile([(500, 31000), (900, 31000), (100, 35000), (1500, 35000), (500, 0)])
    run = cruise_run(alt)
    assert run is not None
    s, e = run
    assert s >= 1500 and e <= 3000  # the 35k plateau only, not joined to the 31k one
    assert e - s > 1400


def test_lower_plateau_wins_when_longer_if_high_enough() -> None:
    alt = profile([(500, 33000), (2500, 33000), (100, 35000), (700, 35000), (500, 0)])
    s, e = cruise_run(alt) or (0, 0)
    assert s >= 500 and e <= 3000 and e - s > 2400


def test_short_plateau_returns_none() -> None:
    alt = profile([(600, 30000), (400, 30000), (600, 0)])  # 400 < 512
    assert cruise_run(alt) is None
    p = CruiseMaskParams(min_len=300)
    assert cruise_run(alt, p) is not None


def test_noisy_plateau_fails_std_threshold() -> None:
    alt = profile([(600, 30000), (2000, 30000), (600, 0)], noise=80.0)
    assert cruise_run(alt) is None
    # same profile is accepted when the noise is below the threshold
    assert cruise_run(profile([(600, 30000), (2000, 30000), (600, 0)], noise=20.0)) is not None


def test_flat_ground_run_is_not_cruise() -> None:
    assert cruise_run(np.full(3000, 100.0)) is None
    assert cruise_run(np.zeros(10)) is None


@pytest.mark.parametrize(
    ("a", "b", "iou"),
    [((0, 10), (0, 10), 1.0), ((0, 10), (5, 15), 5 / 15), ((0, 10), None, 0.0), (None, None, 1.0)],
)
def test_interval_iou(a: tuple[int, int] | None, b: tuple[int, int] | None, iou: float) -> None:
    assert interval_iou(a, b) == pytest.approx(iou)


def test_run_mask() -> None:
    m = run_mask(6, (2, 4))
    assert m.tolist() == [False, False, True, True, False, False]
    assert not run_mask(3, None).any()
