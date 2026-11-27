"""Tiny synthetic N-CMAPSS files with the real HDF5 key layout (test helpers)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

W_VAR = ["alt", "Mach", "TRA", "T2"]
XS_VAR = [
    "T24",
    "T30",
    "T48",
    "T50",
    "P15",
    "P2",
    "P21",
    "P24",
    "Ps30",
    "P40",
    "P50",
    "Nf",
    "Nc",
    "Wf",
]
XV_VAR = [
    "T40",
    "P30",
    "P45",
    "W21",
    "W22",
    "W25",
    "W31",
    "W32",
    "W48",
    "W50",
    "SmFan",
    "SmLPC",
    "SmHPC",
    "phi",
]
T_VAR = [f"{c}_{k}_mod" for c in ("fan", "LPC", "HPC", "HPT", "LPT") for k in ("eff", "flow")]
A_VAR = ["unit", "cycle", "Fc", "hs"]


def flight_altitude(n: int, cruise_alt: float, rng: np.random.Generator) -> np.ndarray:
    """Climb (25%), cruise plateau (50%), descent (25%) with a little noise."""
    c = n // 4
    alt = np.concatenate(
        [
            np.linspace(0, cruise_alt, c),
            np.full(n - 2 * c, cruise_alt),
            np.linspace(cruise_alt, 0, c),
        ]
    )
    return alt + rng.normal(0, 3.0, n)


def synthetic_split(
    units: list[int], n_cycles: int, rows: int, seed: int, onset: int = 6
) -> dict[str, np.ndarray]:
    """Arrays W, X_s, X_v, T, Y, A for some units; sensors drift after the hs onset."""
    rng = np.random.default_rng(seed)
    mix = rng.normal(size=(4, 14))
    parts: dict[str, list[np.ndarray]] = {k: [] for k in ("W", "X_s", "X_v", "T", "Y", "A")}
    for i, u in enumerate(units):
        life = n_cycles + i  # different lives
        fc = 1 + (i % 3)
        for cyc in range(1, life + 1):
            n = rows + int(rng.integers(-rows // 4, rows // 4))
            if cyc == 2:
                n = rows // 3  # one short flight per unit
            alt = flight_altitude(n, 30000.0 + 1000 * fc, rng)
            mach = 0.2 + 0.6 * alt / 35000 + rng.normal(0, 0.005, n)
            tra = 60 + 20 * (alt < 1000) + rng.normal(0, 0.5, n)
            t2 = 520 - 60 * alt / 35000 + rng.normal(0, 0.2, n)
            w = np.stack([alt, mach, tra, t2], axis=1)
            wz = (w - w.mean(0)) / (w.std(0) + 1e-9)
            hs = 1.0 if cyc < onset else 0.0
            deg = max(0, cyc - onset) * 0.05
            x = np.tanh(wz @ mix) + deg + rng.normal(0, 0.01, (n, 14))
            parts["W"].append(w)
            parts["X_s"].append(x)
            parts["X_v"].append(rng.normal(size=(n, 14)))
            parts["T"].append(np.full((n, 10), -deg))
            parts["Y"].append(np.full((n, 1), life - cyc, dtype=np.int64))
            parts["A"].append(np.tile([u, cyc, fc, hs], (n, 1)).astype(np.float64))
    return {k: np.concatenate(v) for k, v in parts.items()}


def write_synthetic_ncmapss(
    path: Path,
    dev_units: list[int],
    test_units: list[int],
    n_cycles: int = 12,
    rows: int = 600,
    seed: int = 0,
) -> Path:
    """Write a synthetic ``N-CMAPSS_*.h5`` with ``*_dev``, ``*_test`` and ``*_var`` keys."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for split, units, s in (("dev", dev_units, seed), ("test", test_units, seed + 1)):
            for k, v in synthetic_split(units, n_cycles, rows, s).items():
                f.create_dataset(f"{k}_{split}", data=v)
        for k, names in (
            ("W", W_VAR),
            ("X_s", XS_VAR),
            ("X_v", XV_VAR),
            ("T", T_VAR),
            ("A", A_VAR),
        ):
            f.create_dataset(f"{k}_var", data=np.array(names, dtype="S"))
    return path
