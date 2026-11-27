"""End-to-end train -> evaluate on synthetic data in a throwaway project root."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from aerosentinel import evaluate as ev
from aerosentinel import train as tr
from aerosentinel.data import prepare as prep
from aerosentinel.runs import CONFIG_DIR

FAST = {
    "lgbm": ["model.params.n_estimators=30", "model.early_stopping_rounds=5"],
    "cnn1d": ["model.train.max_epochs=2"],
    "lstm": ["model.train.max_epochs=2"],
    "ridge": [],
    "constant": [],
}


def _cfg(model: str, extra: list[str] | None = None) -> DictConfig:
    with initialize_config_dir(str(CONFIG_DIR), version_base="1.3"):
        return compose(
            "config",
            overrides=[
                f"experiment=cmapss-fd002_{model}_cap125",
                "data.window=10",
                "seeds=[0,1]",
                "mlflow.tracking_uri=null",
                *FAST[model],
                *(extra or []),
            ],
        )


@pytest.mark.parametrize("model", ["constant", "ridge", "lgbm", "cnn1d", "lstm"])
def test_train_then_evaluate(
    synthetic_root: Path, monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    monkeypatch.setattr(tr, "_log_mlflow", lambda cfg, meta: None)

    def no_test(*a: object, **k: object) -> None:
        raise AssertionError("train must never load official test data")

    monkeypatch.setattr(prep, "load_test", no_test)
    for seed in (0, 1):
        cfg = _cfg(model, [f"seed={seed}"])
        rd = tr.train(cfg)
        assert rd.is_relative_to(synthetic_root)
    monkeypatch.undo()
    monkeypatch.setenv("AEROSENTINEL_ROOT", str(synthetic_root))

    doc = ev.evaluate(_cfg(model))
    out = synthetic_root / "reports" / "results" / f"cmapss-fd002_{model}_cap125.json"
    assert json.loads(out.read_text()) == json.loads(json.dumps(doc))
    assert doc["seeds"] == [0, 1] and len(doc["per_seed"]) == 2
    assert doc["n_test_units"] == 12
    assert doc["n_padded_test_units"] >= 1  # synthetic test units can be < window long
    for key in ("test_capped", "test_uncapped", "val"):
        assert set(doc["summary"][key]) == {"rmse", "mae", "score_sum", "score_mean", "phm21"}
    assert doc["leakage_alarm"]["triggered"] is False  # FD002, alarm is FD001-only
    assert (doc["dataset"], doc["model"], doc["setting"]) == ("cmapss-fd002", model, "cap125")


def test_stale_run_is_rejected(synthetic_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tr, "_log_mlflow", lambda cfg, meta: None)
    for seed in (0, 1):
        tr.train(_cfg("constant", [f"seed={seed}"]))
    with pytest.raises(ev.StaleRunError):
        ev.evaluate(_cfg("constant", ["data.val_frac=0.3"]))


def test_leakage_alarm_rule() -> None:
    low = [{"seed": 0, "test_capped": {"rmse": 10.5}}, {"seed": 1, "test_capped": {"rmse": 12}}]
    assert ev.leakage_alarm("cmapss-fd001", low) == {
        "threshold_rmse": 11.0,
        "triggered": True,
        "seeds": [0],
    }
    assert ev.leakage_alarm("cmapss-fd002", low)["triggered"] is False
