"""Hydra experiment configs follow the SPEC §4 naming scheme."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from aerosentinel.models import MODEL_NAMES

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
STEMS = sorted(p.stem for p in (CONFIGS / "experiment").glob("*.yaml"))
NAME_RE = re.compile(
    r"^(cmapss-fd00[1-4]|ncmapss-ds0[1-7]|ncmapss-pooled)_([a-z0-9]+)_([a-z0-9-]+)$"
)


def test_phase1_grid_exists() -> None:
    for i in range(1, 5):
        for m in ("constant", "ridge", "lgbm", "cnn1d", "lstm"):
            assert f"cmapss-fd00{i}_{m}_cap125" in STEMS


@pytest.mark.parametrize("stem", STEMS)
def test_experiment_name_equals_stem_and_matches_config(stem: str) -> None:
    m = NAME_RE.match(stem)
    assert m, f"{stem} does not follow <dataset>_<model>_<setting>"
    with initialize_config_dir(str(CONFIGS), version_base="1.3"):
        cfg = compose("config", overrides=[f"experiment={stem}"])
    assert cfg.name == stem
    dataset, model, setting = m.groups()
    assert cfg.data.dataset == dataset
    assert cfg.model.name == model and model in MODEL_NAMES
    assert f"cap{cfg.data.cap}" in setting.split("-")
    assert list(cfg.seeds) == [0, 1, 2]
    if dataset in ("cmapss-fd002", "cmapss-fd004"):
        assert cfg.data.n_regimes == 6
    elif dataset.startswith("cmapss-"):
        assert cfg.data.n_regimes == 1
    else:  # N-CMAPSS: the setting name encodes every run-defining data choice
        parts = setting.split("-")
        assert parts[0] == "single"  # single-task (multi-task arrives in Phase 3)
        assert {"full": "full_flight", "cruise": "cruise_only"}[parts[1]] == cfg.data.setting
        assert ("res" in parts) == bool(cfg.data.residuals) != ("nores" in parts)
        assert f"ds{cfg.data.downsample_factor}" in parts
        assert cfg.data.family == "ncmapss"
