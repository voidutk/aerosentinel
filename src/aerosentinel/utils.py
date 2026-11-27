"""Shared utilities: seeding and project paths."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT_ENV_VAR = "AEROSENTINEL_ROOT"


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Seed every random number generator the project uses.

    Parameters
    ----------
    seed : int
        Seed applied to ``random``, NumPy's global RNG and torch (CPU and CUDA).
        It is also exported as ``PYTHONHASHSEED`` for child processes.
    deterministic : bool, default False
        If True, ask torch for deterministic kernels (``warn_only=True``) and
        disable cuDNN autotuning. Slower, but bitwise-reproducible where supported.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    import torch  # lazy: keeps `import aerosentinel` fast

    torch.manual_seed(seed)  # also seeds all CUDA devices
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def project_root() -> Path:
    """Locate the repository root.

    Resolution order: the ``AEROSENTINEL_ROOT`` environment variable, then the
    nearest ancestor of this module that contains ``pyproject.toml`` (editable
    install / source checkout), then the current working directory (e.g. a
    non-editable install on Kaggle run from the repo root).

    Returns
    -------
    pathlib.Path
        Absolute path to the project root.
    """
    env = os.environ.get(ROOT_ENV_VAR)
    if env:
        return Path(env).expanduser().resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd().resolve()


@dataclass(frozen=True)
class Paths:
    """Canonical project locations. Pure value object: creates no directories."""

    root: Path

    @property
    def configs(self) -> Path:
        """Hydra configs and tracked data checksums."""
        return self.root / "configs"

    @property
    def checksums(self) -> Path:
        """Tracked SHA256 + size file for downloaded archives and .h5 files."""
        return self.configs / "data_checksums.txt"

    @property
    def data_raw(self) -> Path:
        """Raw downloads; written only by ``scripts/download_data.py``."""
        return self.root / "data" / "raw"

    @property
    def data_processed(self) -> Path:
        """Derived per-unit arrays and window indices (gitignored)."""
        return self.root / "data" / "processed"

    @property
    def reports(self) -> Path:
        """Reports directory."""
        return self.root / "reports"

    @property
    def results(self) -> Path:
        """Machine-written results JSONs, ``<dataset>_<model>_<setting>.json``."""
        return self.reports / "results"

    @property
    def figures(self) -> Path:
        """Generated figures."""
        return self.reports / "figures"


def get_paths(root: Path | None = None) -> Paths:
    """Return the project :class:`Paths`.

    Parameters
    ----------
    root : pathlib.Path or None, default None
        Explicit project root. If None, :func:`project_root` is used.

    Returns
    -------
    Paths
        Path bundle anchored at ``root``.
    """
    return Paths(root=(root if root is not None else project_root()).resolve())
