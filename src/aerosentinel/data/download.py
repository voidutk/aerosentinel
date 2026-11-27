"""Dataset downloads: resumable HTTP, nested-zip extraction and checksum bookkeeping.

Only ``scripts/download_data.py`` calls into this module, so it is the only writer of
``data/raw/``. Every file written there gets a ``sha256  size  path`` line in the tracked
``configs/data_checksums.txt``.
"""

from __future__ import annotations

import hashlib
import io
import re
import shutil
import sys
import urllib.request
import zipfile
from collections.abc import Callable, Iterable
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


# --- N-CMAPSS (SPEC §1.2) -----------------------------------------------------------------

NCMAPSS_URL = (
    "https://phm-datasets.s3.amazonaws.com/NASA/"
    "17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2.zip"
)
NCMAPSS_ARCHIVE_NAME = "ncmapss.zip"
NCMAPSS_INNER_ZIP = "data_set.zip"
#: Subsets extracted: DS02 for modelling, DS01–DS07 pooled for conformal (SPEC §6).
NCMAPSS_PATTERN = re.compile(r"^N-CMAPSS_DS0[1-7](-\d+)?\.h5$")
NCMAPSS_N_FILES = 7
MIN_FREE_BYTES = 50 * 10**9


class DownloadSizeError(RuntimeError):
    """A local archive's size does not match the server's Content-Length."""


def head_content_length(url: str, *, timeout: float = 60.0) -> int:
    """Return the ``Content-Length`` of ``url`` from an HTTP HEAD request.

    Parameters
    ----------
    url : str
        HTTP(S) URL.
    timeout : float, default 60
        Socket timeout in seconds.

    Returns
    -------
    int
        Size in bytes announced by the server.
    """
    req = urllib.request.Request(
        url, method="HEAD", headers={"User-Agent": "aerosentinel-downloader"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (fixed https URL)
        length = resp.headers.get("Content-Length")
    if length is None:
        raise DownloadSizeError(f"{url}: no Content-Length in HEAD response")
    return int(length)


def check_free_disk(path: Path, min_free: int = MIN_FREE_BYTES) -> int:
    """Raise ``OSError`` if the filesystem holding ``path`` has less than ``min_free`` bytes.

    Returns
    -------
    int
        Free bytes.
    """
    probe = path
    while not probe.exists():
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    if free < min_free:
        raise OSError(f"only {free / 1e9:.1f} GB free at {probe}; need >= {min_free / 1e9:.0f} GB")
    return free


def _basename(name: str) -> str:
    return name.rstrip("/").rsplit("/", 1)[-1]


def stream_extract_nested(
    outer_path: Path,
    inner_basename: str,
    pattern: re.Pattern[str],
    out_dir: Path,
    *,
    log: Callable[[str], None] = print,
) -> list[tuple[Path, str, int]]:
    """Extract members matching ``pattern`` from a zip nested inside ``outer_path``.

    The inner zip is never written to disk: it is opened as a (seekable) stream of the outer
    member. Members are extracted in ``header_offset`` order, so the inflater only ever seeks
    forward after reading the central directory (about two sequential passes). Each member is
    written to a ``.part`` file while its SHA256 is computed, its CRC is checked by
    :mod:`zipfile` at end of stream, and only then is it renamed into place.

    Parameters
    ----------
    outer_path : pathlib.Path
        Outer archive.
    inner_basename : str
        Basename of the nested zip member (e.g. ``"data_set.zip"``).
    pattern : re.Pattern
        Regex on member basenames selecting the files to extract.
    out_dir : pathlib.Path
        Flat destination directory.
    log : callable, default print
        Progress sink.

    Returns
    -------
    list of (path, sha256, size)
        Extracted files in extraction order.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out: list[tuple[Path, str, int]] = []
    with zipfile.ZipFile(outer_path) as outer:
        inner_name = find_member(outer.namelist(), inner_basename)
        with outer.open(inner_name) as inner_stream, zipfile.ZipFile(inner_stream) as inner:
            infos = inner.infolist()
            log(f"inner {inner_basename}: {len(infos)} members")
            for i in infos:
                log(f"  {i.file_size:>14d}  {i.compress_size:>14d}  {i.filename}")
            wanted = sorted(
                (i for i in infos if pattern.match(_basename(i.filename))),
                key=lambda i: i.header_offset,
            )
            for info in wanted:
                target = out_dir / _basename(info.filename)
                part = target.with_suffix(target.suffix + ".part")
                h = hashlib.sha256()
                size = 0
                with inner.open(info) as src, part.open("wb") as dst:
                    while chunk := src.read(_CHUNK * 8):
                        h.update(chunk)
                        dst.write(chunk)
                        size += len(chunk)
                if size != info.file_size:
                    raise zipfile.BadZipFile(f"{info.filename}: wrote {size} != {info.file_size}")
                part.rename(target)
                log(f"extracted {target.name} ({size} B, CRC ok)")
                out.append((target, h.hexdigest(), size))
    return out


def download_ncmapss(
    paths: Paths | None = None,
    *,
    url: str = NCMAPSS_URL,
    keep_zip: bool = False,
    check_remote: bool = True,
    log: Callable[[str], None] = print,
) -> list[Path]:
    """Download N-CMAPSS, stream-extract DS01–DS07 and record checksums (SPEC §1.2).

    Steps: HEAD Content-Length, free-disk check, resumable download (skipped when the local
    zip already has the announced size), stream extraction of the ``.h5`` files through the
    nested ``data_set.zip``, SHA256 + size of the zip and each ``.h5`` into
    ``configs/data_checksums.txt``, then deletion of the zip unless ``keep_zip``.

    Parameters
    ----------
    paths : Paths or None
        Project paths (defaults to :func:`aerosentinel.utils.get_paths`).
    url : str
        Archive URL.
    keep_zip : bool, default False
        Keep the archive after extraction (deviation from SPEC §1.2 step 6).
    check_remote : bool, default True
        Issue the HEAD request (tests pass False).
    log : callable, default print
        Progress sink.

    Returns
    -------
    list of pathlib.Path
        The extracted ``.h5`` files.
    """
    paths = paths or get_paths()
    raw = paths.data_raw / "ncmapss"
    archive = raw / NCMAPSS_ARCHIVE_NAME
    known = read_checksums(paths.checksums)

    def rel(p: Path) -> str:
        return p.relative_to(paths.root).as_posix()

    existing = sorted(p for p in raw.glob("*.h5") if NCMAPSS_PATTERN.match(p.name))
    if len(existing) == NCMAPSS_N_FILES and all(rel(p) in known for p in existing):
        for p in existing:
            status = record_checksum(paths.checksums, rel(p), sha256_file(p), p.stat().st_size)
            log(f"{status:9s} {rel(p)}")
        log("all DS01–DS07 .h5 files present and verified; nothing to do")
        return existing

    if check_remote:
        length = head_content_length(url)
        log(f"HEAD Content-Length: {length} B")
        if archive.is_file() and archive.stat().st_size != length:
            raise DownloadSizeError(
                f"{rel(archive)} is {archive.stat().st_size} B but the server announces "
                f"{length} B; refusing to use it"
            )
    free = check_free_disk(raw)
    log(f"free disk: {free / 1e9:.1f} GB")
    if archive.is_file():
        log(f"{rel(archive)} present ({archive.stat().st_size} B); skipping download")
    else:
        download_resumable(url, archive)
    status = record_checksum(
        paths.checksums, rel(archive), sha256_file(archive), archive.stat().st_size
    )
    log(f"{status:9s} {rel(archive)}")

    extracted = stream_extract_nested(archive, NCMAPSS_INNER_ZIP, NCMAPSS_PATTERN, raw, log=log)
    if len(extracted) != NCMAPSS_N_FILES:
        raise FileNotFoundError(
            f"expected {NCMAPSS_N_FILES} DS01–DS07 .h5 files, found "
            f"{[p.name for p, _, _ in extracted]}"
        )
    for p, sha, size in extracted:
        status = record_checksum(paths.checksums, rel(p), sha, size)
        log(f"{status:9s} {sha}  {size}  {rel(p)}")
    if keep_zip:
        log(f"keeping {rel(archive)} (--keep-zip; deviation from SPEC §1.2 step 6)")
    else:
        archive.unlink()
        log(f"deleted {rel(archive)}")
    return sorted(p for p, _, _ in extracted)
