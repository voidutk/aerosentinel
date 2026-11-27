"""Dataset downloads: resumable HTTP, nested-zip extraction and checksum bookkeeping.

Only ``scripts/download_data.py`` calls into this module, so it is the only writer of
``data/raw/``. Every file written there gets a ``sha256  size  path`` line in the tracked
``configs/data_checksums.txt``.
"""

from __future__ import annotations

import hashlib
import io
import shutil
import sys
import urllib.request
import zipfile
from collections.abc import Iterable
from pathlib import Path

from aerosentinel.utils import Paths, get_paths

CMAPSS_URL = (
    "https://phm-datasets.s3.amazonaws.com/NASA/"
    "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
)
CMAPSS_ARCHIVE_NAME = "cmapss_archive.zip"
CMAPSS_SUBSETS = ("FD001", "FD002", "FD003", "FD004")
CMAPSS_FILES = tuple(
    f"{kind}_{fd}.txt" for fd in CMAPSS_SUBSETS for kind in ("train", "test", "RUL")
)

_CHUNK = 1 << 20


class ChecksumMismatchError(RuntimeError):
    """A file on disk does not match its recorded checksum."""


def sha256_file(path: Path) -> str:
    """Return the hex SHA256 of a file, read in 1 MiB chunks.

    Parameters
    ----------
    path : pathlib.Path
        File to hash.

    Returns
    -------
    str
        Lowercase hex digest.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def read_checksums(path: Path) -> dict[str, tuple[str, int]]:
    """Parse a checksum file into ``{relpath: (sha256, size)}``; comments are ignored.

    Parameters
    ----------
    path : pathlib.Path
        Checksum file with lines ``sha256  size  relpath``.

    Returns
    -------
    dict
        Mapping from relative path to ``(sha256, size_bytes)``.
    """
    out: dict[str, tuple[str, int]] = {}
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        sha, size, rel = line.split(maxsplit=2)
        out[rel] = (sha, int(size))
    return out


def record_checksum(checksum_file: Path, rel: str, sha: str, size: int) -> str:
    """Add a checksum line, or verify an existing one.

    Parameters
    ----------
    checksum_file : pathlib.Path
        Tracked checksum file (header lines are preserved).
    rel : str
        Path relative to the project root, e.g. ``data/raw/cmapss/train_FD001.txt``.
    sha, size : str, int
        Observed SHA256 and size.

    Returns
    -------
    str
        ``"added"`` if a new line was appended, ``"verified"`` if it matched.

    Raises
    ------
    ChecksumMismatchError
        If ``rel`` is already recorded with a different SHA256 or size.
    """
    known = read_checksums(checksum_file)
    if rel in known:
        if known[rel] != (sha, size):
            raise ChecksumMismatchError(
                f"{rel}: recorded {known[rel]}, observed {(sha, size)}. "
                "The upstream file changed or the download is corrupt."
            )
        return "verified"
    text = checksum_file.read_text() if checksum_file.is_file() else ""
    if text and not text.endswith("\n"):
        text += "\n"
    checksum_file.write_text(text + f"{sha}  {size}  {rel}\n")
    return "added"


def download_resumable(url: str, dest: Path, *, timeout: float = 60.0) -> Path:
    """Download ``url`` to ``dest`` with resume support via a ``.part`` file.

    Parameters
    ----------
    url : str
        HTTP(S) URL.
    dest : pathlib.Path
        Final path. Skipped if it already exists.
    timeout : float, default 60
        Socket timeout in seconds.

    Returns
    -------
    pathlib.Path
        ``dest``.
    """
    if dest.is_file():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    offset = part.stat().st_size if part.is_file() else 0
    req = urllib.request.Request(url, headers={"User-Agent": "aerosentinel-downloader"})
    if offset:
        req.add_header("Range", f"bytes={offset}-")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (fixed https URL)
        status = getattr(resp, "status", 200)
        mode = "ab" if offset and status == 206 else "wb"
        with part.open(mode) as f:
            shutil.copyfileobj(resp, f, _CHUNK)
    part.rename(dest)
    return dest


def find_member(names: Iterable[str], basename: str) -> str:
    """Return the unique archive member whose basename equals ``basename``.

    Parameters
    ----------
    names : iterable of str
        Archive member names.
    basename : str
        File name to look for (case-sensitive).

    Returns
    -------
    str
        Full member name.

    Raises
    ------
    FileNotFoundError
        If no or more than one member matches.
    """
    hits = [n for n in names if n.rstrip("/").rsplit("/", 1)[-1] == basename]
    if len(hits) != 1:
        raise FileNotFoundError(f"expected exactly one {basename!r} in archive, found {hits}")
    return hits[0]


def extract_cmapss_files(archive: Path, out_dir: Path) -> list[Path]:
    """Extract the 12 C-MAPSS text files, looking inside a nested ``CMAPSSData.zip``.

    Parameters
    ----------
    archive : pathlib.Path
        Outer archive as downloaded.
    out_dir : pathlib.Path
        Destination directory (flat).

    Returns
    -------
    list of pathlib.Path
        Written files, in :data:`CMAPSS_FILES` order.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as outer:
        names = outer.namelist()
        try:
            find_member(names, "train_FD001.txt")
            inner = outer
        except FileNotFoundError:
            nested = find_member(names, "CMAPSSData.zip")
            inner = zipfile.ZipFile(io.BytesIO(outer.read(nested)))
        written = []
        inner_names = inner.namelist()
        for fname in CMAPSS_FILES:
            data = inner.read(find_member(inner_names, fname))
            target = out_dir / fname
            target.write_bytes(data)
            written.append(target)
    return written


def download_cmapss(paths: Paths | None = None, *, url: str = CMAPSS_URL) -> list[Path]:
    """Download C-MAPSS from the primary URL, extract it, and record checksums.

    Parameters
    ----------
    paths : Paths or None
        Project paths (defaults to :func:`aerosentinel.utils.get_paths`).
    url : str
        Primary archive URL (no mirror fallback; see SPEC §1.1).

    Returns
    -------
    list of pathlib.Path
        The archive followed by the 12 extracted text files.
    """
    paths = paths or get_paths()
    raw = paths.data_raw / "cmapss"
    archive = download_resumable(url, raw / CMAPSS_ARCHIVE_NAME)
    files = [archive, *extract_cmapss_files(archive, raw)]
    for f in files:
        rel = f.relative_to(paths.root).as_posix()
        status = record_checksum(paths.checksums, rel, sha256_file(f), f.stat().st_size)
        print(f"{status:9s} {rel}", file=sys.stderr)
    return files
