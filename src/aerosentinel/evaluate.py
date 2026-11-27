"""Final evaluation: ``python -m aerosentinel.evaluate experiment=<stem>``.

The only code that loads official C-MAPSS test units. For every seed in ``cfg.seeds`` it
loads the saved run, predicts the last (edge-padded if short) window of each test unit and
scores it against capped (primary) and uncapped truth, overall and split into edge-padded
and full-length test units. It writes
``reports/results/<stem>.json`` with per-seed metrics and mean ± std (ddof=1).
"""

from __future__ import annotations

import json
import pickle
import sys
from typing import Any

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from aerosentinel.data.cmapss import cap_truth
from aerosentinel.data.prepare import Preprocessor, prepare_test
from aerosentinel.metrics import point_metrics
from aerosentinel.models import build_model
from aerosentinel.runs import CONFIG_DIR, cfg_dict, config_hash, git_info, raw_dir, run_dir
from aerosentinel.utils import get_paths

#: SPEC §3: a capped FD001 RMSE below this means "hunt for leakage before celebrating".
LEAKAGE_ALARM_RMSE = 11.0
LEAKAGE_EXIT_CODE = 3


class StaleRunError(RuntimeError):
    """A saved run was trained with a different config than the one being evaluated."""


def summarize(per_seed: list[dict[str, Any]], key: str) -> dict[str, dict[str, float]]:
    """Mean and sample std (ddof=1) of every metric under ``per_seed[i][key]``."""
    metrics = per_seed[0][key].keys()
    out = {}
    for m in metrics:
        v = np.array([s[key][m] for s in per_seed], dtype=np.float64)
        out[m] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0}
    return out


def leakage_alarm(dataset: str, per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    """Flag any capped FD001 test RMSE below :data:`LEAKAGE_ALARM_RMSE`."""
    low = [s["seed"] for s in per_seed if s["test_capped"]["rmse"] < LEAKAGE_ALARM_RMSE]
    triggered = dataset == "cmapss-fd001" and bool(low)
    return {"threshold_rmse": LEAKAGE_ALARM_RMSE, "triggered": triggered, "seeds": low}


def padding_breakdown(y_capped: np.ndarray, pred: np.ndarray, padded: np.ndarray) -> dict[str, Any]:
    """Capped-truth metrics on padded vs full-length test units (None when a group is empty).

    Parameters
    ----------
    y_capped, pred : numpy.ndarray
        Capped truth and predictions, one per test unit.
    padded : numpy.ndarray of bool
        True for units shorter than the window (edge-padded).
    """
    out: dict[str, Any] = {}
    for key, mask in (("padded", padded), ("not_padded", ~padded)):
        n = int(mask.sum())
        out[key] = {"n_units": n, **point_metrics(y_capped[mask], pred[mask])} if n else None
    return out


def evaluate(cfg: DictConfig) -> dict[str, Any]:
    """Evaluate every seed of an experiment and write its results JSON.

    Returns
    -------
    dict
        The results document that was written.
    """
    paths = get_paths()
    name, cap, window = str(cfg.name), cfg.data.cap, int(cfg.data.window)
    expected_hash = config_hash(cfg)
    per_seed: list[dict[str, Any]] = []
    test_info: dict[str, Any] = {}
    for seed in [int(s) for s in cfg.seeds]:
        rd = run_dir(paths, name, seed)
        meta = json.loads((rd / "train_meta.json").read_text())
        if meta["config_hash"] != expected_hash:
            raise StaleRunError(
                f"{rd}: trained with config {meta['config_hash']}, "
                f"evaluating {expected_hash}; retrain it."
            )
        with (rd / "preprocessor.pkl").open("rb") as f:
            pre: Preprocessor = pickle.load(f)  # noqa: S301 - our own run artifact
        model = build_model(cfg.model, target_scale=float(cap) if cap is not None else 125.0)
        model.load(rd)
        # The one place in the code base allowed to read official test files.
        test = prepare_test(raw_dir(cfg, paths), str(cfg.data.subset), pre, window, allow_test=True)
        pred = model.predict(test.x, test.regime_last).astype(np.float64)
        y_capped = cap_truth(test.rul_true, None if cap is None else int(cap))
        y_raw = cap_truth(test.rul_true, None)
        pd.DataFrame(
            {
                "unit": test.units,
                "length": test.lengths,
                "n_padded": test.n_padded,
                "rul_true": test.rul_true,
                "rul_true_capped": y_capped,
                "pred": pred,
            }
        ).to_csv(rd / "test_predictions.csv", index=False)
        per_seed.append(
            {
                "seed": seed,
                "val": meta["val"],
                "test_capped": point_metrics(y_capped, pred),
                "test_uncapped": point_metrics(y_raw, pred),
                "test_capped_by_padding": padding_breakdown(y_capped, pred, test.n_padded > 0),
                "train_git": meta.get("git"),
                "n_params": meta["n_params"],
                "train_seconds": meta["train_seconds"],
                "fit_info": meta["fit_info"],
            }
        )
        test_info = {
            "n_test_units": len(test.units),
            "n_padded_test_units": int((test.n_padded > 0).sum()),
            "min_test_length": int(test.lengths.min()),
            "n_train_units": len(meta["train_units"]),
            "n_val_units": len(meta["val_units"]),
            "n_train_windows": meta["n_train_windows"],
            "channels": meta["channels"],
            "sensors_dropped": meta["sensors_dropped"],
            "sensors_dropped_by_rule": meta.get("sensors_dropped_by_rule"),
            "n_inputs": len(meta["input_names"]),
        }

    dataset, model_name, setting = name.split("_", 2)
    doc: dict[str, Any] = {
        "experiment": name,
        "dataset": dataset,
        "model": model_name,
        "setting": setting,
        "written_by": "aerosentinel.evaluate",
        "protocol": {
            "split": f"unit-level; val = {cfg.data.val_frac:.0%} of train-file units "
            f"(split_seed={cfg.split_seed}); preprocessing fit on train-split units only",
            "test": "last window per official test unit; edge-padded (first cycle repeated) "
            "when the unit has fewer cycles than the window",
            "truth": f"capped at {cap} (primary) and uncapped (secondary)",
            "score": "Saxena score, d = pred - true; score_sum is the C-MAPSS convention",
            "std": "sample std over seeds (ddof=1)",
        },
        "config_hash": expected_hash,
        "config": cfg_dict(cfg),
        "git": git_info(paths.root),
        "seeds": [int(s) for s in cfg.seeds],
        **test_info,
        "summary": {
            **{k: summarize(per_seed, k) for k in ("val", "test_capped", "test_uncapped")},
            "test_capped_by_padding": {
                g: summarize([s["test_capped_by_padding"] for s in per_seed], g)
                if per_seed[0]["test_capped_by_padding"][g]
                else None
                for g in ("padded", "not_padded")
            },
        },
        "per_seed": per_seed,
        "leakage_alarm": leakage_alarm(dataset, per_seed),
    }
    doc["config"].pop("seed", None)
    paths.results.mkdir(parents=True, exist_ok=True)
    out = paths.results / f"{name}.json"
    out.write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n")
    s = doc["summary"]["test_capped"]["rmse"]
    print(f"[evaluate] {name}: capped RMSE {s['mean']:.2f} ± {s['std']:.2f} -> {out}")
    return doc


@hydra.main(version_base="1.3", config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint; exits with code 3 if the leakage alarm fires (C-MAPSS)."""
    if str(cfg.data.family) == "ncmapss":
        from aerosentinel.pipelines.ncmapss import evaluate_ncmapss

        evaluate_ncmapss(cfg)
        return
    doc = evaluate(cfg)
    if doc["leakage_alarm"]["triggered"]:
        print(
            f"[evaluate] LEAKAGE ALARM: capped FD001 RMSE < {LEAKAGE_ALARM_RMSE}", file=sys.stderr
        )
        sys.exit(LEAKAGE_EXIT_CODE)


if __name__ == "__main__":
    main()
