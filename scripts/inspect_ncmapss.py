"""Print HDF5 keys, shapes, dtypes and variable names of the extracted N-CMAPSS files.

Usage: ``uv run python scripts/inspect_ncmapss.py [DS02 ...]`` (default: all DS01–DS07).
"""

from __future__ import annotations

import argparse
import json

from aerosentinel.data.ncmapss import describe_h5
from aerosentinel.utils import get_paths


def main(argv: list[str] | None = None) -> int:
    """Describe each requested subset."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("datasets", nargs="*", default=[f"DS0{i}" for i in range(1, 8)])
    args = ap.parse_args(argv)
    raw = get_paths().data_raw / "ncmapss"
    for ds in args.datasets:
        (path,) = sorted(raw.glob(f"N-CMAPSS_{ds}*.h5"))
        info = describe_h5(path)
        print(f"=== {path.name} ({path.stat().st_size / 1e9:.2f} GB)")
        for key, d in info["datasets"].items():
            print(f"  {key:10s} {str(d['shape']):18s} {d['dtype']:8s} chunks={d['chunks']}")
        for group, names in info["var_names"].items():
            print(f"  {group}_var ({len(names)}): {json.dumps(names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
