"""Hydra training entrypoint: ``python -m aerosentinel.train experiment=<stem> seed=<s>``.

Fits preprocessing and the model on train-split units (validation units for early stopping
and selection), saves the run to ``outputs/runs/<stem>/seed_<s>/`` and logs to MLflow.
It never reads official test files; :mod:`aerosentinel.evaluate` does that.
"""

from __future__ import annotations

import json
import os
import pickle
import time
import warnings
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig

from aerosentinel.metrics import point_metrics
from aerosentinel.models import build_model
from aerosentinel.models.base import predict_windows
from aerosentinel.runs import CONFIG_DIR, cfg_dict, config_hash, prepare_dev_from_cfg, run_dir
from aerosentinel.utils import get_paths, seed_everything


def _log_mlflow(cfg: DictConfig, meta: dict[str, Any]) -> None:
    """Best-effort MLflow logging (results JSONs, not MLflow, are the source of numbers)."""
    try:
        import mlflow

        uri = cfg.mlflow.tracking_uri or f"sqlite:///{get_paths().root / 'mlflow.db'}"
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(str(cfg.mlflow.experiment))
        with mlflow.start_run(run_name=f"{cfg.name}_seed{cfg.seed}"):
            mlflow.set_tags({"experiment": cfg.name, "config_hash": meta["config_hash"]})
            mlflow.log_params({"seed": cfg.seed, "model": cfg.model.name, **dict(cfg.data)})
            mlflow.log_metrics({f"val_{k}": v for k, v in meta["val"].items()})
            mlflow.log_metric("n_params", meta["n_params"])
    except Exception as e:  # noqa: BLE001 - logging must never fail a run
        warnings.warn(f"MLflow logging failed: {e}", stacklevel=2)


def train(cfg: DictConfig) -> Path:
    """Train one seed of an experiment and save it.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        Composed Hydra config.

    Returns
    -------
    pathlib.Path
        The run directory.
    """
    seed = int(cfg.seed)
    seed_everything(seed, deterministic=bool(cfg.deterministic))
    dev = prepare_dev_from_cfg(cfg)
    cap = cfg.data.cap
    model = build_model(cfg.model, target_scale=float(cap) if cap is not None else 125.0)
    t0 = time.perf_counter()
    fit_info = model.fit(dev.train, dev.val, seed)
    train_seconds = time.perf_counter() - t0
    val_metrics = point_metrics(dev.val.window_labels(), predict_windows(model, dev.val))

    rd = run_dir(get_paths(), str(cfg.name), seed)
    rd.mkdir(parents=True, exist_ok=True)
    model.save(rd)
    with (rd / "preprocessor.pkl").open("wb") as f:
        pickle.dump(dev.pre, f)
    meta = {
        "experiment": str(cfg.name),
        "seed": seed,
        "config_hash": config_hash(cfg),
        "config": cfg_dict(cfg),
        "train_units": dev.train.units,
        "val_units": dev.val.units,
        "n_train_windows": len(dev.train.index),
        "n_val_windows": len(dev.val.index),
        "channels": dev.pre.channels,
        "sensors_dropped": dev.pre.selector.dropped_,
        "input_names": model.input_names(),
        "val": val_metrics,
        "fit_info": fit_info,
        "n_params": model.n_params(),
        "train_seconds": train_seconds,
        "threads": os.environ.get("OMP_NUM_THREADS"),
    }
    (rd / "train_meta.json").write_text(json.dumps(meta, indent=2))
    _log_mlflow(cfg, meta)
    print(f"[train] {cfg.name} seed={seed} val_rmse={val_metrics['rmse']:.3f} -> {rd}")
    return rd


@hydra.main(version_base="1.3", config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint."""
    train(cfg)


if __name__ == "__main__":
    main()
