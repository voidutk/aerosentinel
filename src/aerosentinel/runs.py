"""Run bookkeeping shared by train and evaluate: run directories, config hash, git info."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from aerosentinel.data.prepare import DevData, prepare_dev
from aerosentinel.utils import Paths, get_paths

CONFIG_DIR = get_paths().configs


def cfg_dict(cfg: DictConfig) -> dict[str, Any]:
    """Resolved config as a plain dict, without the ``hydra`` node."""
    d = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(d, dict)
    d.pop("hydra", None)
    return {str(k): v for k, v in d.items()}


def config_hash(cfg: DictConfig) -> str:
    """Hash of everything that defines the experiment (all keys except ``seed``)."""
    d = cfg_dict(cfg)
    d.pop("seed", None)
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


def run_dir(paths: Paths, name: str, seed: int) -> Path:
    """Directory holding one trained seed of an experiment (gitignored ``outputs/``)."""
    return paths.root / "outputs" / "runs" / name / f"seed_{seed}"


def raw_dir(cfg: DictConfig, paths: Paths | None = None) -> Path:
    """Raw-data directory of the configured dataset family."""
    return (paths or get_paths()).data_raw / str(cfg.data.family)


def prepare_dev_from_cfg(cfg: DictConfig) -> DevData:
    """Run :func:`aerosentinel.data.prepare.prepare_dev` with the config's data settings."""
    d = cfg.data
    return prepare_dev(
        raw_dir(cfg),
        str(d.subset),
        cap=None if d.cap is None else int(d.cap),
        window=int(d.window),
        stride=int(d.stride),
        val_frac=float(d.val_frac),
        split_seed=int(cfg.split_seed),
        n_regimes=int(d.n_regimes),
        var_threshold=float(d.var_threshold),
        max_distinct=int(d.max_distinct),
    )


#: Generated outputs: changes here do not make the tree dirty (``evaluate`` writes results
#: JSONs and ``make_tables`` rewrites README tables while a batch of runs is in progress).
GENERATED_PATHS: tuple[str, ...] = ("reports/results", "README.md")


def git_info(root: Path) -> dict[str, Any]:
    """Current commit SHA and whether tracked code or config had uncommitted changes.

    Untracked files (logs, scratch output) and the generated outputs in
    :data:`GENERATED_PATHS` do not count as dirty: they cannot change what the committed
    code computes.
    """

    def _git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=root, capture_output=True, text=True, check=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    return {
        "sha": _git("rev-parse", "HEAD") or None,
        "dirty": bool(
            _git(
                "status",
                "--porcelain",
                "--untracked-files=no",
                "--",
                ".",
                *(f":(exclude){p}" for p in GENERATED_PATHS),
            )
        ),
    }
