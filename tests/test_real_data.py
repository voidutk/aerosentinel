"""Checks on the real C-MAPSS files (slow; skipped when data/raw/cmapss is absent)."""

from __future__ import annotations

import json

import pytest

from aerosentinel.data.cmapss import RegimeNormalizer, load_train
from aerosentinel.data.download import read_checksums, sha256_file
from aerosentinel.utils import get_paths

RAW = get_paths().data_raw / "cmapss"
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not (RAW / "train_FD001.txt").is_file(), reason="C-MAPSS not downloaded"),
]

#: (train units, test units) as found in the files. SPEC §1.1 lists FD004 as 248/249;
#: the files have 249 train and 248 test units.
UNITS = {"FD001": (100, 100), "FD002": (260, 259), "FD003": (100, 100), "FD004": (249, 248)}


def test_checksums_match_files() -> None:
    paths = get_paths()
    known = read_checksums(paths.checksums)
    cmapss = {k: v for k, v in known.items() if k.startswith("data/raw/cmapss/")}
    assert len(cmapss) == 13
    for rel, (sha, size) in cmapss.items():
        f = paths.root / rel
        assert f.stat().st_size == size
        assert sha256_file(f) == sha


@pytest.mark.parametrize("subset", list(UNITS))
def test_unit_counts(subset: str) -> None:
    """Train units from the file; test units from the results JSONs.

    Test files are read only by ``aerosentinel.evaluate`` (which checks ids are 1..N and
    match the RUL file), so the test-unit count is taken from what it wrote.
    """
    assert load_train(RAW, subset)["unit"].nunique() == UNITS[subset][0]
    docs = list(get_paths().results.glob(f"cmapss-{subset.lower()}_*.json"))
    assert docs, f"no results JSONs for {subset}"
    for p in docs:
        doc = json.loads(p.read_text())
        assert doc["n_train_units"] + doc["n_val_units"] == UNITS[subset][0]
        assert doc["n_test_units"] == UNITS[subset][1]


@pytest.mark.parametrize("subset", ["FD002", "FD004"])
def test_kmeans_recovers_six_operating_conditions(subset: str) -> None:
    df = load_train(RAW, subset)
    reg = RegimeNormalizer(n_regimes=6, seed=0).fit(df).regimes(df)
    cond = df[["setting1", "setting2", "setting3"]].round({"setting1": 0, "setting2": 2})
    key = cond.astype(str).agg("|".join, axis=1)
    # Each regime maps to exactly one rounded operating condition, and vice versa.
    pairs = set(zip(reg.tolist(), key.tolist(), strict=True))
    assert len(pairs) == 6
    assert len({r for r, _ in pairs}) == 6 and len({k for _, k in pairs}) == 6
