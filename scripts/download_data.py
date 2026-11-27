"""Download raw datasets into data/raw/ (the ONLY writer of data/raw/).

Usage: ``uv run python scripts/download_data.py {cmapss,ncmapss}``

C-MAPSS is implemented (Phase 1). N-CMAPSS arrives in Phase 2 (see docs/PHASES.md).
"""

from __future__ import annotations

import argparse
import sys

from aerosentinel.data.download import download_cmapss


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to the dataset downloader."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", choices=["cmapss", "ncmapss"])
    args = parser.parse_args(argv)
    if args.dataset == "cmapss":
        files = download_cmapss()
        print(f"C-MAPSS: {len(files)} files in place, checksums recorded.", file=sys.stderr)
        return 0
    print("download_data.py ncmapss: not implemented until Phase 2.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
