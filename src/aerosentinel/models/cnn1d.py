"""1D-CNN in the style of Li, Ding & Sun (2018): convolutions along time, per sensor."""

from __future__ import annotations

import torch
from torch import nn


class CNN1D(nn.Module):
    """Li et al. 2018 DCNN.

    Input [B, T, C] is treated as a 1-channel image [B, 1, T, C]. Kernels span time only
    (k x 1), so each sensor is filtered along time and sensors are mixed only by the FC layer.

    Parameters
    ----------
    n_channels, window : int
        Input channels C and window length T.
    n_conv : int, default 4
        Number of (filters, k x 1) conv layers before the final 1-filter layer.
    filters, kernel : int, default 10, 10
        Filters and time-kernel of those layers.
    final_kernel : int, default 3
        Time-kernel of the last (1-filter) conv layer.
    fc : int, default 100
        Hidden FC width.
    dropout : float, default 0.5
        Dropout before the FC layer.
    """

    def __init__(
        self,
        n_channels: int,
        window: int,
        n_conv: int = 4,
        filters: int = 10,
        kernel: int = 10,
        final_kernel: int = 3,
        fc: int = 100,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        c_in = 1
        for _ in range(n_conv):
            layers += [nn.Conv2d(c_in, filters, (kernel, 1), padding="same"), nn.Tanh()]
            c_in = filters
        layers += [nn.Conv2d(c_in, 1, (final_kernel, 1), padding="same"), nn.Tanh()]
        self.conv = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(window * n_channels, fc),
            nn.Tanh(),
            nn.Linear(fc, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map windows [B, T, C] to RUL [B] (in target-scaled units)."""
        out: torch.Tensor = self.head(self.conv(x.unsqueeze(1)))
        return out.squeeze(-1)
