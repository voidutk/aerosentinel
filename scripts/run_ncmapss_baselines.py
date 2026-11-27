"""Produce the Phase 2 N-CMAPSS numbers: train all seeds, evaluate, rebuild README tables.

Usage::

    uv run python scripts/run_ncmapss_baselines.py [--jobs 2] [--threads 8] [--skip-train]

Runs every ``configs/experiment/ncmapss-ds02_lgbm_*.yaml`` (LightGBM on DS02 at downsample 10:
{full_flight, cruise_only} x {W + X_s + residuals, W + X_s}). Each (experiment, seed) trains in
its own process; ``python -m aerosentinel.evaluate`` then writes
``reports/results/<stem>.json`` and ``scripts/make_tables.py`` updates the README.
Requires ``scripts/prepare_ncmapss.py --datasets DS02`` first.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from hydra import compose, initialize_config_dir

from aerosentinel.utils import get_paths

ROOT = get_paths().root


def experiments(pattern: str) -> list[str]:
    """Experiment stems matching ``pattern`` in ``configs/experiment``."""
    stems = sorted(p.stem for p in (ROOT / "configs" / "experiment").glob(f"{pattern}.yaml"))
    if not stems:
        raise FileNotFoundError(pattern)
    return stems


def seeds_of(stem: str) -> list[int]:
    """Seeds the experiment config declares (``cfg.seeds``)."""
    with initialize_config_dir(str(ROOT / "configs"), version_base="1.3"):
        cfg = compose("config", overrides=[f"experiment={stem}"])
    return [int(s) for s in cfg.seeds]


def _run(cmd: list[str], log: Path, threads: int) -> int:
    env = {**os.environ, "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads)}
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as f:
        return subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT).returncode


def main(argv: list[str] | None = None) -> int:
    """Train, evaluate and tabulate; return a process exit code."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pattern", default="ncmapss-ds02_lgbm_*")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args(argv)
    stems = experiments(args.pattern)
    py = [sys.executable, "-m"]
    logs = ROOT / "outputs" / "logs"
    failed: list[str] = []
    if not args.skip_train:
        tasks = [(st, s) for st in stems for s in seeds_of(st)]

        def train(task: tuple[str, int]) -> tuple[str, int, int]:
            st, s = task
            cmd = [*py, "aerosentinel.train", f"experiment={st}", f"seed={s}"]
            rc = _run(cmd, logs / f"{st}_seed{s}.log", args.threads)
            print(f"[train] {st} seed={s} rc={rc}", flush=True)
            return st, s, rc

        with ThreadPoolExecutor(args.jobs) as pool:
            for st, s, rc in pool.map(train, tasks):
                if rc != 0:
                    failed.append(f"train {st} seed={s} (see {logs / f'{st}_seed{s}.log'})")
        if failed:
            print("\n".join(["FAILED:", *failed]), file=sys.stderr)
            return 1
    for st in stems:
        log = logs / f"{st}_eval.log"
        rc = _run([*py, "aerosentinel.evaluate", f"experiment={st}"], log, args.threads)
        print(log.read_text().strip().splitlines()[-1], flush=True)
        if rc != 0:
            failed.append(f"evaluate {st}")
    rc = subprocess.run([sys.executable, str(ROOT / "scripts" / "make_tables.py")], cwd=ROOT)
    if rc.returncode != 0:
        failed.append("make_tables")
    if failed:
        print("\n".join(["FAILED:", *failed]), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
