"""Synthetic C-MAPSS data in the real file format, plus a script loader (test helpers)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd

from aerosentinel.data.cmapss import COLUMNS, SENSORS

ROOT = Path(__file__).resolve().parents[1]

#: The six real FD002/FD004 operating conditions (setting1, setting2, setting3).
CONDITIONS = np.array(
    [
        [0, 0.0, 100],
        [10, 0.25, 100],
        [20, 0.7, 100],
        [25, 0.62, 60],
        [35, 0.84, 100],
        [42, 0.84, 100],
    ],
    dtype=np.float64,
)


def load_script(name: str) -> ModuleType:
    """Import ``scripts/<name>.py`` as a module."""
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"scripts_{name}", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def synthetic_cmapss(
    n_units: int,
    *,
    n_regimes: int,
    seed: int,
    min_len: int = 40,
    max_len: int = 90,
    truncate: bool = False,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Synthetic trajectories with regime offsets, a degradation trend and constant sensors.

    s1 and s5 are constant within each regime; every other sensor drifts with life fraction.
    If ``truncate``, each unit is cut before failure (test-style) and the true RUL at its
    last cycle is returned.
    """
    rng = np.random.default_rng(seed)
    rows, ruls = [], []
    offsets = rng.normal(0, 5, size=(len(CONDITIONS), len(SENSORS)))
    for u in range(1, n_units + 1):
        life = int(rng.integers(min_len, max_len))
        n = int(rng.integers(5, life)) if truncate else life
        ruls.append(life - n)
        for c in range(1, n + 1):
            k = int(rng.integers(0, n_regimes))
            s = CONDITIONS[k] + rng.normal(0, 1e-4, 3) * np.array([1, 1, 0])
            frac = c / life
            x = 100 + offsets[k] + 3 * frac**2 + rng.normal(0, 0.1, len(SENSORS))
            x[[0, 4]] = 100 + offsets[k, [0, 4]]  # s1, s5 constant within a regime
            rows.append([u, c, *s, *x])
    df = pd.DataFrame(rows, columns=list(COLUMNS))
    return df.astype({"unit": "int64", "cycle": "int64"}), np.array(ruls, dtype=np.int64)


def write_cmapss_txt(df: pd.DataFrame, path: Path) -> None:
    """Write a frame in the raw C-MAPSS format (space-separated, trailing spaces)."""
    lines = []
    for row in df.itertuples(index=False):
        vals = [str(int(row[0])), str(int(row[1]))] + [f"{v:.4f}" for v in row[2:]]
        lines.append(" ".join(vals) + "  \n")
    path.write_text("".join(lines))
