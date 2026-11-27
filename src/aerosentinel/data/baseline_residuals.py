"""Per-engine healthy baseline W -> X_s and residual channels (SPEC §1.3 item 5).

For each engine ``u`` (train, val, calib and test alike) a small MLP ``f_u`` maps the four
flight descriptors W to the 14 measured sensors X_s. It is fit on **that engine's own first
5 cycles only**: no health state, no labels, no other engine. Because the baseline uses only
the engine's own past, it is legal at inference time, and it absorbs the engine's unknown
initial health. The residual channels are ``(X_s - f_u(W)) / sigma_u``, where ``sigma_u`` is
the per-sensor std of X_s over the same first 5 cycles.

Residuals on cycles 1-5 are in-sample (near zero by construction), so models that use them
predict from cycle 6 onward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

F32 = NDArray[np.float32]
F64 = NDArray[np.float64]

BASELINE_CYCLES = 5


@dataclass(frozen=True)
class BaselineConfig:
    """MLP and fitting settings of the healthy baseline (fixed, not tuned)."""

    n_cycles: int = BASELINE_CYCLES
    hidden: int = 64
    max_rows: int = 20_000
    steps: int = 2_000
    lr: float = 3e-3
    weight_decay: float = 1e-5
    seed: int = 0


@dataclass
class EngineBaseline:
    """A fitted per-engine baseline: input/output standardization plus the MLP weights."""

    cycles: list[int]
    w_mean: F64
    w_std: F64
    x_mean: F64
    x_std: F64
    w_min: F64
    w_max: F64
    n_fit_rows: int
    fit_mse: float
    state: dict[str, torch.Tensor] = field(repr=False)
    hidden: int = 64

    def module(self) -> nn.Module:
        """The MLP in standardized units (W_std -> X_s_std), weights loaded, eval mode."""
        net = _mlp(len(self.w_mean), len(self.x_mean), self.hidden)
        net.load_state_dict(self.state)
        return net.eval()


def _mlp(n_in: int, n_out: int, hidden: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(n_in, hidden),
        nn.SiLU(),
        nn.Linear(hidden, hidden),
        nn.SiLU(),
        nn.Linear(hidden, n_out),
    )


def baseline_cycles(cycle: NDArray[Any], n_cycles: int = BASELINE_CYCLES) -> list[int]:
    """The engine's first ``n_cycles`` distinct cycle numbers (its own earliest flights)."""
    return [int(c) for c in np.unique(np.asarray(cycle))[:n_cycles]]


def _train_mlp(w: F64, x: F64, cfg: BaselineConfig) -> tuple[dict[str, torch.Tensor], float]:
    """Full-batch Adam on standardized (w, x); returns weights and final train MSE."""
    gen = torch.Generator().manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    net = _mlp(w.shape[1], x.shape[1], cfg.hidden)
    for p in net.parameters():  # deterministic init from our own generator
        if p.dim() > 1:
            nn.init.kaiming_uniform_(p, a=5**0.5, generator=gen)
        else:
            nn.init.zeros_(p)
    wt = torch.as_tensor(w, dtype=torch.float32)
    xt = torch.as_tensor(x, dtype=torch.float32)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.steps)
    loss = torch.tensor(0.0)
    for _ in range(cfg.steps):
        opt.zero_grad()
        loss = nn.functional.mse_loss(net(wt), xt)
        loss.backward()  # type: ignore[no-untyped-call]
        opt.step()
        sched.step()
    with torch.no_grad():
        mse = float(nn.functional.mse_loss(net(wt), xt))
    return {k: v.detach().clone() for k, v in net.state_dict().items()}, mse


def fit_engine_baseline(
    w: NDArray[Any], x_s: NDArray[Any], cycle: NDArray[Any], cfg: BaselineConfig | None = None
) -> EngineBaseline:
    """Fit one engine's healthy baseline on its own first ``cfg.n_cycles`` cycles.

    The row selection happens here: callers pass the engine's whole trajectory and only the
    rows of its first ``n_cycles`` distinct cycles are used (subsampled evenly to at most
    ``max_rows``). There is deliberately no argument for health state, labels or other
    engines.

    Parameters
    ----------
    w : numpy.ndarray
        Flight descriptors [N, 4] of one engine.
    x_s : numpy.ndarray
        Measured sensors [N, 14] of the same engine.
    cycle : numpy.ndarray
        Cycle number of each row [N].
    cfg : BaselineConfig or None
        Fitting settings (defaults if None).

    Returns
    -------
    EngineBaseline
        The fitted baseline.
    """
    cfg = cfg or BaselineConfig()
    cyc = baseline_cycles(cycle, cfg.n_cycles)
    rows = np.flatnonzero(np.isin(np.asarray(cycle), cyc))
    if rows.size == 0:
        raise ValueError("engine has no rows")
    if rows.size > cfg.max_rows:
        rows = rows[np.linspace(0, rows.size - 1, cfg.max_rows).round().astype(np.int64)]
    wf = np.asarray(w[rows], dtype=np.float64)
    xf = np.asarray(x_s[rows], dtype=np.float64)
    w_mean, w_std = wf.mean(0), np.maximum(wf.std(0), 1e-6)
    x_mean, x_std = xf.mean(0), np.maximum(xf.std(0), 1e-6)
    state, mse = _train_mlp((wf - w_mean) / w_std, (xf - x_mean) / x_std, cfg)
    all_fit = np.asarray(w[np.isin(np.asarray(cycle), cyc)], dtype=np.float64)
    return EngineBaseline(
        cycles=cyc,
        w_mean=w_mean,
        w_std=w_std,
        x_mean=x_mean,
        x_std=x_std,
        w_min=all_fit.min(0),
        w_max=all_fit.max(0),
        n_fit_rows=int(rows.size),
        fit_mse=mse,
        state=state,
        hidden=cfg.hidden,
    )


def residuals(
    base: EngineBaseline, w: NDArray[Any], x_s: NDArray[Any], chunk: int = 500_000
) -> F32:
    """Residual channels ``(X_s - f_u(W)) / sigma_u`` [N, 14] for one engine (chunked)."""
    net = base.module()
    out = np.empty((len(w), len(base.x_mean)), dtype=np.float32)
    with torch.no_grad():
        for a in range(0, len(w), chunk):
            wz = (np.asarray(w[a : a + chunk], dtype=np.float64) - base.w_mean) / base.w_std
            xz = (np.asarray(x_s[a : a + chunk], dtype=np.float64) - base.x_mean) / base.x_std
            pred = net(torch.as_tensor(wz, dtype=torch.float32)).numpy().astype(np.float64)
            out[a : a + chunk] = (xz - pred).astype(np.float32)
    return out


def extrapolation_share(base: EngineBaseline, w: NDArray[Any], cycle: NDArray[Any]) -> float:
    """Share of rows after the baseline cycles whose W leaves the cycles-1-5 [min, max] box.

    A row counts as extrapolated if any of the 4 descriptors is outside the per-dimension
    range seen in the baseline cycles. Returns NaN if the engine has no later rows.
    """
    later = ~np.isin(np.asarray(cycle), base.cycles)
    if not later.any():
        return float("nan")
    wl = np.asarray(w[later], dtype=np.float64)
    out = ((wl < base.w_min) | (wl > base.w_max)).any(axis=1)
    return float(out.mean())
