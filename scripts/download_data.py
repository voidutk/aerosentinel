"""Download raw datasets into data/raw/ (the ONLY writer of data/raw/).

Usage: ``uv run python scripts/download_data.py {cmapss,ncmapss} [--keep-zip]``

``ncmapss`` follows SPEC §1.2: HEAD Content-Length, free-disk check, resumable download,
stream extraction of DS01–DS07 through the nested ``data_set.zip``, checksums, and deletion
of the archive unless ``--keep-zip``.
"""

from __future__ import annotations

import argparse
import sys

from aerosentinel.data.download import download_cmapss, download_ncmapss


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to the dataset downloader."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", choices=["cmapss", "ncmapss"])
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="ncmapss: keep the archive after extraction (SPEC §1.2 says delete it)",
    )
    args = parser.parse_args(argv)
    if args.dataset == "cmapss":
        files = download_cmapss()
        _log(f"C-MAPSS: {len(files)} files in place, checksums recorded.")
        return 0
    h5 = download_ncmapss(keep_zip=args.keep_zip, log=_log)
    _log(f"N-CMAPSS: {len(h5)} .h5 files in place, checksums recorded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
