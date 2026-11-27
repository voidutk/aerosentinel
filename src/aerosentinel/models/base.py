"""Common model interface and allowed-input bookkeeping."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from aerosentinel.data.cmapss import SENSORS
from aerosentinel.data.prepare import WindowData

F32 = NDArray[np.float32]

#: Base names a model input may be derived from on C-MAPSS. Settings are absorbed by the
#: regime normalizer (the regime id is the only setting-derived input, LightGBM only).
ALLOWED_INPUT_BASES: frozenset[str] = frozenset(SENSORS) | {"regime_id"}
#: Columns that must never reach a model (label leakage or identity / life-length priors).
BANNED_INPUTS: frozenset[str] = frozenset(
    {"unit", "cycle", "RUL", "rul", "setting1", "setting2", "setting3"}
)


def feature_base(name: str) -> str:
    """Return the source column of a feature name (``"s7__slope"`` -> ``"s7"``)."""
    return name.split("__", 1)[0]


class RULModel(Protocol):
    """Interface every Phase 1 model implements."""

    name: str

    def fit(self, train: WindowData, val: WindowData, seed: int) -> dict[str, Any]:
        """Fit on train windows (val for early stopping / selection). Return fit info."""
        ...

    def predict(self, x: F32, regime_last: NDArray[np.int64]) -> F32:
        """Predict RUL for windows ``x`` [B, T, C]."""
        ...

    def input_names(self) -> list[str]:
        """Names of the model's input features / channels (checked by the leak test)."""
        ...

    def n_params(self) -> int:
        """Number of fitted parameters (trees: number of leaves)."""
        ...

    def save(self, run_dir: Path) -> None:
        """Persist the fitted model in ``run_dir``."""
        ...

    def load(self, run_dir: Path) -> None:
        """Restore a model saved by :meth:`save`."""
        ...


def predict_windows(model: RULModel, data: WindowData, batch_size: int = 4096) -> F32:
    """Predict every indexed window of ``data`` in batches (index order)."""
    out = [model.predict(x, r) for x, _, r in data.batches(batch_size)]
    return np.concatenate(out).astype(np.float32) if out else np.zeros(0, np.float32)
