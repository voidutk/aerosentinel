"""Shared torch training loop (SPEC §5.3) and a :class:`RULModel` wrapper for nets.

AdamW, cosine decay with linear warmup, early stopping on validation RMSE (best weights
restored), gradient clipping. Targets are divided by ``target_scale`` (the RUL cap) during
training and multiplied back at prediction.
"""

from __future__ import annotations

import copy
import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.utils.data import DataLoader

from aerosentinel.data.prepare import WindowData
from aerosentinel.data.windows import WindowDataset
from aerosentinel.metrics import rmse

F32 = NDArray[np.float32]


@dataclass
class TrainConfig:
    """Training hyperparameters (defaults from SPEC §5.3)."""

    lr: float = 3e-4
    weight_decay: float = 1e-2
    batch_size: int = 256
    max_epochs: int = 100
    patience: int = 10
    warmup_frac: float = 0.05
    grad_clip: float = 1.0
    target_scale: float = 125.0


def cosine_with_warmup(total_steps: int, warmup_steps: int) -> Callable[[int], float]:
    """LR multiplier: linear warmup to 1, then cosine decay to 0 at ``total_steps``."""

    def f(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    return f


@torch.no_grad()
def _predict(net: nn.Module, x: F32, scale: float, batch: int = 4096) -> F32:
    net.eval()
    out = [net(torch.from_numpy(x[i : i + batch])) for i in range(0, len(x), batch)]
    return (torch.cat(out).numpy() * scale).astype(np.float32)


@torch.no_grad()
def _predict_data(net: nn.Module, data: WindowData, scale: float) -> F32:
    return np.concatenate([_predict(net, x, scale) for x, _, _ in data.batches()])


def train_net(
    net: nn.Module, train: WindowData, val: WindowData, cfg: TrainConfig, seed: int
) -> dict[str, Any]:
    """Train ``net`` in place; restore the best-validation-RMSE weights.

    Returns
    -------
    dict
        ``best_epoch``, ``best_val_rmse``, ``epochs_run`` and ``train_seconds``.
    """
    ds = WindowDataset(train.arrays, train.labels, train.index)
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, generator=gen)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total = cfg.max_epochs * len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, cosine_with_warmup(total, int(cfg.warmup_frac * total))
    )
    yva = val.window_labels()
    loss_fn = nn.MSELoss()
    best_rmse, best_epoch, best_state = float("inf"), -1, copy.deepcopy(net.state_dict())
    t0 = time.perf_counter()
    epoch = 0
    for epoch in range(cfg.max_epochs):
        net.train()
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(net(xb), yb / cfg.target_scale)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
        val_rmse = rmse(yva, _predict_data(net, val, cfg.target_scale))
        if val_rmse < best_rmse:
            best_rmse, best_epoch = val_rmse, epoch
            best_state = copy.deepcopy(net.state_dict())
        elif epoch - best_epoch >= cfg.patience:
            break
    net.load_state_dict(best_state)
    return {
        "best_epoch": best_epoch,
        "best_val_rmse": best_rmse,
        "epochs_run": epoch + 1,
        "train_seconds": time.perf_counter() - t0,
    }


class TorchModel:
    """:class:`~aerosentinel.models.base.RULModel` wrapper around a torch net factory."""

    def __init__(
        self,
        name: str,
        factory: Callable[[int, int], nn.Module],
        train_cfg: TrainConfig,
    ) -> None:
        self.name = name
        self.factory = factory
        self.train_cfg = train_cfg
        self.net: nn.Module | None = None
        self.channels: list[str] = []
        self.window = 0

    def fit(self, train: WindowData, val: WindowData, seed: int) -> dict[str, Any]:
        """Build the net (after seeding) and train it."""
        self.channels = list(train.channels)
        self.window = train.index.window
        torch.manual_seed(seed)
        self.net = self.factory(len(self.channels), self.window)
        return train_net(self.net, train, val, self.train_cfg, seed)

    def _net(self) -> nn.Module:
        if self.net is None:
            raise RuntimeError("model is not fitted")
        return self.net

    def predict(self, x: F32, regime_last: NDArray[np.int64]) -> F32:
        """Predict RUL for windows [B, T, C] (regime is not an input to nets)."""
        return _predict(self._net(), np.ascontiguousarray(x), self.train_cfg.target_scale)

    def input_names(self) -> list[str]:
        """Input channels."""
        return list(self.channels)

    def n_params(self) -> int:
        """Trainable parameter count."""
        return sum(p.numel() for p in self._net().parameters() if p.requires_grad)

    def save(self, run_dir: Path) -> None:
        """Save the state dict and the shape metadata."""
        torch.save(self._net().state_dict(), run_dir / "net.pt")
        meta = {"channels": self.channels, "window": self.window, "train": asdict(self.train_cfg)}
        (run_dir / "net.json").write_text(json.dumps(meta))

    def load(self, run_dir: Path) -> None:
        """Rebuild the net and load a state dict saved by :meth:`save`."""
        meta = json.loads((run_dir / "net.json").read_text())
        self.channels, self.window = list(meta["channels"]), int(meta["window"])
        self.net = self.factory(len(self.channels), self.window)
        self.net.load_state_dict(torch.load(run_dir / "net.pt", weights_only=True))
        self.net.eval()
