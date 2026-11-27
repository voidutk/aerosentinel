"""Download raw datasets into data/raw/ (the ONLY writer of data/raw/).

Usage: ``uv run python scripts/download_data.py {cmapss,ncmapss}``

Stub: C-MAPSS is implemented in Phase 1, N-CMAPSS in Phase 2 (see docs/PHASES.md).
"""

from __future__ import annotations

import argparse
import sys

_PHASE = {"cmapss": 1, "ncmapss": 2}


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to the dataset downloader."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", choices=sorted(_PHASE))
    args = parser.parse_args(argv)
    print(
        f"download_data.py {args.dataset}: not implemented until Phase {_PHASE[args.dataset]}.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
