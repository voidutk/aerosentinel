"""LSTM baseline (SPEC §5.1 item 3): 2 layers, 64 hidden units."""

from __future__ import annotations

import torch
from torch import nn


class LSTMRegressor(nn.Module):
    """Stacked LSTM with a linear head on the last time step.

    Parameters
    ----------
    n_channels : int
        Input channels C.
    hidden : int, default 64
        Hidden size.
    layers : int, default 2
        Number of LSTM layers.
    dropout : float, default 0.2
        Dropout between LSTM layers and before the head.
    """

    def __init__(
        self, n_channels: int, hidden: int = 64, layers: int = 2, dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            n_channels, hidden, num_layers=layers, batch_first=True, dropout=dropout
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map windows [B, T, C] to RUL [B] (in target-scaled units)."""
        out, _ = self.lstm(x)
        y: torch.Tensor = self.head(out[:, -1])
        return y.squeeze(-1)
