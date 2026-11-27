"""Shared fixtures: a throwaway project root with synthetic C-MAPSS files."""

from __future__ import annotations

from pathlib import Path

import pytest

from synth import synthetic_cmapss, write_cmapss_txt


@pytest.fixture
def synthetic_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throwaway project root with synthetic FD001 (1 regime) and FD002 (6 regimes)."""
    raw = tmp_path / "data" / "raw" / "cmapss"
    raw.mkdir(parents=True)
    for subset, k, seed in (("FD001", 1, 0), ("FD002", 6, 1)):
        train, _ = synthetic_cmapss(30, n_regimes=k, seed=seed)
        test, rul = synthetic_cmapss(12, n_regimes=k, seed=seed + 100, truncate=True)
        write_cmapss_txt(train, raw / f"train_{subset}.txt")
        write_cmapss_txt(test, raw / f"test_{subset}.txt")
        (raw / f"RUL_{subset}.txt").write_text("".join(f"{r} \n" for r in rul))
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "data_checksums.txt").write_text("# header\n")
    monkeypatch.setenv("AEROSENTINEL_ROOT", str(tmp_path))
    return tmp_path
