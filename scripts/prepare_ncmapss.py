"""Build the N-CMAPSS per-unit store, cruise runs and healthy-baseline residuals.

Usage: ``uv run python scripts/prepare_ncmapss.py [--datasets DS02 ...]``

For each subset and split (dev, test): convert the ``.h5`` into per-unit float32 arrays under
``data/processed/ncmapss/<DS>/<split>/``, cache the cruise run of every flight for the
default mask, and fit each engine's healthy baseline on its own first 5 cycles. Writes
``reports/results/ncmapss-<ds>_data_baseline-residuals.json``.

Test units are read here only for format conversion and for input-only derivations (mask and
baseline use W, X_s and cycle; never Y or hs).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

from aerosentinel.data.baseline_residuals import BaselineConfig
from aerosentinel.data.download import read_checksums
from aerosentinel.data.masks import DEFAULT_MASK
from aerosentinel.data.ncmapss import SPLITS, UnitStore, convert_split
from aerosentinel.data.ncmapss_derive import compute_cruise_runs, compute_residuals
from aerosentinel.runs import git_info
from aerosentinel.utils import get_paths


def main(argv: list[str] | None = None) -> int:
    """Convert and derive the requested subsets."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--datasets", nargs="+", default=["DS02"])
    ap.add_argument("--skip-convert", action="store_true", help="reuse converted arrays")
    args = ap.parse_args(argv)
    paths = get_paths()
    raw = paths.data_raw / "ncmapss"
    root = paths.data_processed / "ncmapss"
    sums = read_checksums(paths.checksums)
    cfg = BaselineConfig()
    for ds in args.datasets:
        (h5,) = sorted(raw.glob(f"N-CMAPSS_{ds}*.h5"))
        sha = sums.get(h5.relative_to(paths.root).as_posix(), (None, 0))[0]
        report: dict[str, object] = {
            "experiment": f"ncmapss-{ds.lower()}_data_baseline-residuals",
            "dataset": f"ncmapss-{ds.lower()}",
            "model": "data",
            "setting": "baseline-residuals",
            "written_by": "scripts/prepare_ncmapss.py",
            "source": h5.name,
            "baseline_config": asdict(cfg),
            "mask_params": DEFAULT_MASK.to_dict(),
            "git": git_info(paths.root),
            "definition": {
                "residual": "(X_s - f_u(W)) / sigma_u, f_u = per-engine MLP fit on the "
                "engine's own first 5 cycles (no hs, no labels, no other engine)",
                "w_extrapolation_share": "share of rows after cycle 5 with any W descriptor "
                "outside the per-dimension [min, max] seen in cycles 1-5",
            },
            "splits": {},
        }
        for split in SPLITS:
            t0 = time.perf_counter()
            out = root / ds / split
            if not args.skip_convert:
                man = convert_split(h5, split, out, source_sha256=sha)
                print(f"[prepare] {ds} {split}: {man['n_rows']} rows, {len(man['units'])} units")
            store = UnitStore(root, ds, split, allow_test=True)  # input-only derivations
            cruise = compute_cruise_runs(store, DEFAULT_MASK)
            res = compute_residuals(store, cfg)
            report["splits"][split] = {  # type: ignore[index]
                "units": {
                    str(u): {
                        **res[u],
                        "n_flights": store.manifest["units"][str(u)]["n_flights"],
                        "n_flights_with_cruise_run": cruise[u],
                    }
                    for u in store.units
                }
            }
            print(f"[prepare] {ds} {split}: derived in {time.perf_counter() - t0:.0f}s")
        paths.results.mkdir(parents=True, exist_ok=True)
        dest = paths.results / f"{report['experiment']}.json"
        dest.write_text(json.dumps(report, indent=2) + "\n")
        print(f"[prepare] wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
