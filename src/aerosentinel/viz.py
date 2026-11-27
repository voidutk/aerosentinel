"""Shared matplotlib style for report figures (static PNGs, light surface).

Colors follow the dataviz reference palette: categorical slots in fixed order (blue,
orange, aqua; at most three series share a panel), text in text inks, recessive grid.
"""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#ffffff"
TEXT = "#191919"
TEXT_2 = "#666666"
GRID = "#e6e6e6"
SERIES = ("#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647")
#: Sequential blue ramp, light -> dark (ordinal use starts at step 250).
BLUES = ("#84b9f3", "#519ced", "#257de0", "#1b64be", "#0f4a96", "#0e3d7e")
NEUTRAL = "#9c9c9c"


def apply_style() -> None:
    """Set rcParams for every figure in this process."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_2,
            "axes.titlecolor": TEXT,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": TEXT_2,
            "ytick.color": TEXT_2,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 2.0,
            "font.size": 9,
            "text.color": TEXT,
            "savefig.dpi": 130,
            "savefig.bbox": "tight",
        }
    )


def save(fig: Any, path: Any) -> None:
    """Save and close a figure."""
    fig.savefig(path)
    plt.close(fig)
