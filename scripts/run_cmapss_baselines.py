"""Produce every Phase 1 C-MAPSS number: train all seeds, evaluate, rebuild README tables.

Usage::

    uv run python scripts/run_cmapss_baselines.py [--jobs 4] [--models lgbm cnn1d] \
        [--subsets fd001] [--skip-train]

Each (experiment, seed) trains in its own process (``python -m aerosentinel.train``) with
``OMP_NUM_THREADS = threads``. Then ``python -m aerosentinel.evaluate`` writes
``reports/results/<stem>.json`` per experiment, and ``scripts/make_tables.py`` updates the
README. Exits non-zero if any run fails or the FD001 leakage alarm fires (SPEC §3).
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
MODELS = ("constant", "ridge", "lgbm", "cnn1d", "lstm")
SUBSETS = ("fd001", "fd002", "fd003", "fd004")
#: Rough relative cost, used only to start the slowest runs first.
_COST = {"cnn1d": 4, "lstm": 2, "lgbm": 1, "ridge": 1, "constant": 0}
_SIZE = {"fd001": 1, "fd002": 2.5, "fd003": 1.2, "fd004": 3}


def experiments(models: list[str], subsets: list[str]) -> list[str]:
    """Experiment stems present in ``configs/experiment`` for the chosen models/subsets."""
    stems = []
    for s in subsets:
        for m in models:
            stem = f"cmapss-{s}_{m}_cap125"
            if not (ROOT / "configs" / "experiment" / f"{stem}.yaml").is_file():
                raise FileNotFoundError(stem)
            stems.append(stem)
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


def init_mlflow() -> None:
    """Create the MLflow sqlite DB once, before parallel workers write to it."""
    try:
        import mlflow

        mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
        mlflow.set_experiment("aerosentinel-cmapss")
    except Exception as e:  # noqa: BLE001
        print(f"warning: MLflow init failed ({e}); runs will warn and continue", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    """Train, evaluate and tabulate; return a process exit code."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    ap.add_argument("--subsets", nargs="+", default=list(SUBSETS), choices=SUBSETS)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 4) // 4))
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args(argv)

    stems = experiments(args.models, args.subsets)
    py = [sys.executable, "-m"]
    logs = ROOT / "outputs" / "logs"
    failed: list[str] = []

    if not args.skip_train:
        init_mlflow()
        tasks = [(st, s) for st in stems for s in seeds_of(st)]
        tasks.sort(key=lambda t: -_COST[t[0].split("_")[1]] * _SIZE[t[0][7:12]])

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

    alarms = []
    for st in stems:
        rc = _run([*py, "aerosentinel.evaluate", f"experiment={st}"], logs / f"{st}_eval.log", 4)
        print((logs / f"{st}_eval.log").read_text().strip().splitlines()[-1], flush=True)
        if rc == 3:
            alarms.append(st)
        elif rc != 0:
            failed.append(f"evaluate {st}")
    rc = subprocess.run([sys.executable, str(ROOT / "scripts" / "make_tables.py")], cwd=ROOT)
    if rc.returncode != 0:
        failed.append("make_tables")
    if alarms:
        print(f"LEAKAGE ALARM (capped FD001 RMSE < 11): {alarms}", file=sys.stderr)
        return 3
    if failed:
        print("\n".join(["FAILED:", *failed]), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
