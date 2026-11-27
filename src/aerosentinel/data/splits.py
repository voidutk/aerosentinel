"""Unit-level splits (SPEC §1.3 item 6). A unit's rows live in exactly one split."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def train_val_units(
    units: Sequence[int], val_frac: float = 0.2, seed: int = 0
) -> tuple[list[int], list[int]]:
    """Split unit ids into disjoint train and validation sets.

    Parameters
    ----------
    units : sequence of int
        Unique unit ids.
    val_frac : float, default 0.2
        Fraction of units held out for validation (rounded, at least 1).
    seed : int, default 0
        Split seed; independent of the model seed so model comparisons are paired.

    Returns
    -------
    (list of int, list of int)
        Sorted train units and sorted validation units.
    """
    u = np.array(sorted(set(units)), dtype=np.int64)
    if len(u) < 2:
        raise ValueError("need at least 2 units to split")
    n_val = max(1, int(round(val_frac * len(u))))
    perm = np.random.default_rng(seed).permutation(len(u))
    val = sorted(int(x) for x in u[perm[:n_val]])
    train = sorted(int(x) for x in u[perm[n_val:]])
    return train, val


def unit_kfold(
    units: Sequence[int], k: int = 5, seed: int = 0
) -> list[tuple[list[int], list[int]]]:
    """K-fold over units: each unit is held out exactly once (for OOF predictions).

    Parameters
    ----------
    units : sequence of int
        Unique unit ids.
    k : int, default 5
        Number of folds.
    seed : int, default 0
        Shuffle seed.

    Returns
    -------
    list of (train_units, heldout_units)
        One pair per fold, both sorted.
    """
    u = np.array(sorted(set(units)), dtype=np.int64)
    if not 2 <= k <= len(u):
        raise ValueError(f"k={k} must be in [2, n_units={len(u)}]")
    perm = np.random.default_rng(seed).permutation(len(u))
    folds = np.array_split(u[perm], k)
    out = []
    for i, held in enumerate(folds):
        train = np.concatenate([f for j, f in enumerate(folds) if j != i])
        out.append((sorted(int(x) for x in train), sorted(int(x) for x in held)))
    return out


def rotation_folds(
    units: Sequence[int], n_val: int = 1, n_calib: int = 1, seed: int = 0
) -> list[tuple[list[int], list[int], list[int]]]:
    """Leave-unit-out rotation into (train, val, calib) folds (SPEC §1.3 item 6).

    Units are shuffled once and split into ``n_folds = n_units // n_val`` groups. Fold ``i``
    validates on group ``i`` and calibrates on group ``i + 1`` (cyclically), so every unit is
    a validation unit exactly once and a calibration unit exactly once, and calibration
    units never train or early-stop that fold's model (their predictions are out-of-fold).
    For DS02's 6 dev units the defaults give 6 folds of 4 train / 1 val / 1 calib.

    Parameters
    ----------
    units : sequence of int
        Unique dev unit ids.
    n_val, n_calib : int, default 1
        Units per validation / calibration group (must be equal; one group size).
    seed : int, default 0
        Shuffle seed (the split seed, independent of model seeds).

    Returns
    -------
    list of (train_units, val_units, calib_units)
        One triple per fold, each sorted.
    """
    if n_val != n_calib:
        raise ValueError("n_val and n_calib must be equal (one group size)")
    u = np.array(sorted(set(units)), dtype=np.int64)
    if len(u) % n_val or len(u) // n_val < 3:
        raise ValueError(f"{len(u)} units cannot form >= 3 groups of {n_val}")
    perm = u[np.random.default_rng(seed).permutation(len(u))]
    groups = [perm[i : i + n_val] for i in range(0, len(u), n_val)]
    out = []
    for i in range(len(groups)):
        val, cal = groups[i], groups[(i + 1) % len(groups)]
        held = set(val.tolist()) | set(cal.tolist())
        train = [int(x) for x in u if int(x) not in held]
        out.append((train, sorted(int(x) for x in val), sorted(int(x) for x in cal)))
    return out
