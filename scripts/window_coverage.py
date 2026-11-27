"""Fraction of DS02 dev cycles with >= 1 window per (setting, downsample, T); fixes T (ds10).

Usage: ``uv run python scripts/window_coverage.py``
Writes ``reports/results/ncmapss-ds02_data_window-coverage.json``.

Resolves the SPEC §1.3 item 3 open point with a rule fixed before any model was trained:
at downsample 10, T is the largest candidate in {64, 128, 256} for which >= 95% of DS02 dev
cycles have at least one window in ``cruise_only`` (the stricter setting). Dev units only.
"""

from __future__ import annotations

import json

from aerosentinel.data.masks import DEFAULT_MASK
from aerosentinel.data.ncmapss import UnitStore
from aerosentinel.data.ncmapss_windows import SETTINGS, cycle_coverage
from aerosentinel.runs import git_info
from aerosentinel.utils import get_paths

WINDOWS = (64, 128, 256)
DOWNSAMPLES = (1, 10)
RULE_SETTING = "cruise_only"
RULE_DOWNSAMPLE = 10
RULE_MIN_FRACTION = 0.95


def main() -> int:
    """Compute the coverage grid and the chosen T."""
    paths = get_paths()
    store = UnitStore(paths.data_processed / "ncmapss", "DS02", "dev")
    rows = []
    for setting in SETTINGS:
        for k in DOWNSAMPLES:
            for t in WINDOWS:
                for first in (1, 6):
                    cov = cycle_coverage(store, store.units, setting, DEFAULT_MASK, t, k, first)
                    rows.append(
                        {
                            "setting": setting,
                            "downsample": k,
                            "window": t,
                            "span_minutes": t * k / 60.0,
                            "min_cycle": first,
                            **cov,
                        }
                    )
                    print(
                        f"{setting:11s} ds{k:<2d} T={t:3d} cycles>={first}: "
                        f"{cov['n_with_window']}/{cov['n_cycles']} = {cov['fraction']:.3f}"
                    )
    ok = [
        r["window"]
        for r in rows
        if r["setting"] == RULE_SETTING
        and r["downsample"] == RULE_DOWNSAMPLE
        and r["min_cycle"] == 1
        and r["fraction"] >= RULE_MIN_FRACTION
    ]
    chosen = max(ok) if ok else None
    doc = {
        "experiment": "ncmapss-ds02_data_window-coverage",
        "dataset": "ncmapss-ds02",
        "model": "data",
        "setting": "window-coverage",
        "written_by": "scripts/window_coverage.py",
        "units": store.units,
        "split": "dev",
        "mask_params": DEFAULT_MASK.to_dict(),
        "note": "1 native sample = 1 s. A cycle counts if its segment (whole flight or its "
        "cruise run) has at least T model samples after keeping every k-th row.",
        "rule": f"largest T in {list(WINDOWS)} with >= {RULE_MIN_FRACTION:.0%} of dev cycles "
        f"having >= 1 window in {RULE_SETTING} at downsample {RULE_DOWNSAMPLE}",
        "chosen_window_ds10": chosen,
        "grid": rows,
        "git": git_info(paths.root),
    }
    out = paths.results / f"{doc['experiment']}.json"
    out.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"chosen T at downsample 10: {chosen} -> {out}")
    return 0 if chosen else 1


if __name__ == "__main__":
    raise SystemExit(main())
