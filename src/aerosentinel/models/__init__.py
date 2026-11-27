"""Models: naive baselines, LightGBM, 1D-CNN, LSTM, and a registry keyed by config."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aerosentinel.models.base import RULModel
from aerosentinel.models.cnn1d import CNN1D
from aerosentinel.models.lgbm import LGBMModel
from aerosentinel.models.lstm import LSTMRegressor
from aerosentinel.models.naive import ConstantRUL, RidgeLastWindow
from aerosentinel.models.trainer import TorchModel, TrainConfig

MODEL_NAMES: tuple[str, ...] = ("constant", "ridge", "lgbm", "cnn1d", "lstm")


def build_model(cfg: Mapping[str, Any], target_scale: float) -> RULModel:
    """Instantiate an unfitted model from its config.

    Parameters
    ----------
    cfg : mapping
        Model config (``configs/model/<name>.yaml``), with ``name`` and model keys.
    target_scale : float
        Scale applied to targets during neural-net training (the RUL cap).

    Returns
    -------
    RULModel
        Unfitted model.
    """
    name = str(cfg["name"])
    if name == "constant":
        return ConstantRUL()
    if name == "ridge":
        return RidgeLastWindow(alphas=list(cfg["alphas"]))
    if name == "lgbm":
        return LGBMModel(dict(cfg["params"]), int(cfg["early_stopping_rounds"]))
    if name in ("cnn1d", "lstm"):
        tc = TrainConfig(**dict(cfg["train"]), target_scale=target_scale)
        arch = dict(cfg["arch"])
        if name == "cnn1d":
            return TorchModel(name, lambda c, t: CNN1D(c, t, **arch), tc)
        return TorchModel(name, lambda c, t: LSTMRegressor(c, **arch), tc)
    raise ValueError(f"unknown model {name!r}; expected one of {MODEL_NAMES}")


__all__ = ["MODEL_NAMES", "RULModel", "build_model"]
