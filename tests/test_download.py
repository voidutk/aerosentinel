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
