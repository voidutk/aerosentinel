"""Download helpers: nested-zip extraction and checksum bookkeeping (no network)."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from aerosentinel.data import download as dl


def _nested_archive(path: Path) -> None:
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as z:
        for name in dl.CMAPSS_FILES:
            z.writestr(name, f"content of {name}\n")
        z.writestr("readme.txt", "ignored")
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(
            "6. Turbofan Engine Degradation Simulation Data Set/CMAPSSData.zip", inner.getvalue()
        )
        z.writestr("6. Turbofan Engine Degradation Simulation Data Set/other.pdf", b"%PDF")


def test_extract_from_nested_zip(tmp_path: Path) -> None:
    archive = tmp_path / "a.zip"
    _nested_archive(archive)
    out = dl.extract_cmapss_files(archive, tmp_path / "raw")
    assert [p.name for p in out] == list(dl.CMAPSS_FILES)
    assert (tmp_path / "raw" / "RUL_FD004.txt").read_text() == "content of RUL_FD004.txt\n"
    assert not (tmp_path / "raw" / "readme.txt").exists()


def test_find_member_requires_unique_match() -> None:
    assert dl.find_member(["a/b/train_FD001.txt", "x.txt"], "train_FD001.txt") == (
        "a/b/train_FD001.txt"
    )
    with pytest.raises(FileNotFoundError):
        dl.find_member(["x.txt"], "train_FD001.txt")
    with pytest.raises(FileNotFoundError):
        dl.find_member(["a/train_FD001.txt", "b/train_FD001.txt"], "train_FD001.txt")


def test_checksum_add_verify_and_mismatch(tmp_path: Path) -> None:
    f = tmp_path / "data_checksums.txt"
    f.write_text("# sha256  size_bytes  path\n")
    assert dl.record_checksum(f, "data/raw/x.txt", "ab" * 32, 10) == "added"
    assert dl.record_checksum(f, "data/raw/x.txt", "ab" * 32, 10) == "verified"
    lines = f.read_text().splitlines()
    assert lines[0].startswith("#") and len(lines) == 2  # header kept, no duplicate line
    assert dl.read_checksums(f) == {"data/raw/x.txt": ("ab" * 32, 10)}
    with pytest.raises(dl.ChecksumMismatchError):
        dl.record_checksum(f, "data/raw/x.txt", "cd" * 32, 10)


def test_sha256_file(tmp_path: Path) -> None:
    p = tmp_path / "f.bin"
    p.write_bytes(b"abc")
    assert dl.sha256_file(p) == ("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")


# --- N-CMAPSS ---------------------------------------------------------------------------

# Real naming: DS01–DS03 carry a version suffix, DS04–DS07 do not.
H5_NAMES = [
    "N-CMAPSS_DS01-005.h5",
    "N-CMAPSS_DS02-006.h5",
    "N-CMAPSS_DS03-012.h5",
    "N-CMAPSS_DS04.h5",
    "N-CMAPSS_DS05.h5",
    "N-CMAPSS_DS06.h5",
    "N-CMAPSS_DS07.h5",
]


def _ncmapss_root(tmp_path: Path, names: list[str]) -> Path:
    """Project root with a fake N-CMAPSS archive: outer zip -> data_set.zip -> .h5 members."""
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("data_set/N-CMAPSS_DS08a-009.h5", b"decoy")  # not extracted
        z.writestr("data_set/N-CMAPSS_DS08d-010.h5", b"decoy")
        for n in names:  # reverse order: extraction must still follow header offsets
            z.writestr(f"data_set/{n}", (n * 1000).encode())
        z.writestr("data_set/Run_to_Failure_Simulation_Under_Real_Flight_Conditions.pdf", b"%")
    raw = tmp_path / "data" / "raw" / "ncmapss"
    raw.mkdir(parents=True)
    with zipfile.ZipFile(raw / dl.NCMAPSS_ARCHIVE_NAME, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("17. Turbofan Engine Degradation Simulation Data Set 2/", b"")
        z.writestr(
            "17. Turbofan Engine Degradation Simulation Data Set 2/data_set.zip", inner.getvalue()
        )
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "data_checksums.txt").write_text("# header\n")
    return tmp_path


@pytest.mark.parametrize("keep_zip", [True, False])
def test_ncmapss_stream_extract_and_checksums(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, keep_zip: bool
) -> None:
    from aerosentinel.utils import get_paths

    root = _ncmapss_root(tmp_path, list(reversed(H5_NAMES)))
    paths = get_paths(root)
    monkeypatch.setattr(dl, "check_free_disk", lambda path: 10**12)
    files = dl.download_ncmapss(paths, keep_zip=keep_zip, check_remote=False, log=lambda _: None)
    raw = root / "data" / "raw" / "ncmapss"
    assert [p.name for p in files] == sorted(H5_NAMES)
    assert (raw / H5_NAMES[1]).read_bytes() == (H5_NAMES[1] * 1000).encode()
    assert not (raw / "N-CMAPSS_DS08a-009.h5").exists()
    assert not (raw / "N-CMAPSS_DS08d-010.h5").exists()
    assert not (raw / "data_set.zip").exists()  # inner zip never written to disk
    assert not list(raw.glob("*.part"))
    known = dl.read_checksums(paths.checksums)
    assert set(known) == {f"data/raw/ncmapss/{n}" for n in [*H5_NAMES, dl.NCMAPSS_ARCHIVE_NAME]}
    assert (raw / dl.NCMAPSS_ARCHIVE_NAME).exists() == keep_zip
    # Second run: everything present and verified, the archive is not needed any more.
    again = dl.download_ncmapss(paths, check_remote=False, log=lambda _: None)
    assert [p.name for p in again] == sorted(H5_NAMES)


def test_ncmapss_missing_subsets_abort(tmp_path: Path) -> None:
    from aerosentinel.utils import get_paths

    root = _ncmapss_root(tmp_path, H5_NAMES[:3])
    with pytest.raises(FileNotFoundError):
        dl.download_ncmapss(get_paths(root), keep_zip=True, check_remote=False, log=lambda _: None)
    assert (root / "data" / "raw" / "ncmapss" / dl.NCMAPSS_ARCHIVE_NAME).exists()


def test_ncmapss_size_mismatch_refuses_local_zip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from aerosentinel.utils import get_paths

    root = _ncmapss_root(tmp_path, H5_NAMES)
    monkeypatch.setattr(dl, "head_content_length", lambda url: 123)
    with pytest.raises(dl.DownloadSizeError):
        dl.download_ncmapss(get_paths(root), keep_zip=True, log=lambda _: None)


def test_free_disk_check(tmp_path: Path) -> None:
    assert dl.check_free_disk(tmp_path / "missing" / "dir", min_free=0) > 0
    with pytest.raises(OSError):
        dl.check_free_disk(tmp_path, min_free=10**18)
