"""Build every README markdown table from ``reports/results/*.json``.

Each table lives in the README between ``<!-- results:<key>:start -->`` and
``<!-- results:<key>:end -->``. Later phases add builders to :data:`TABLES`.

Usage::

    uv run python scripts/make_tables.py          # rewrite README in place
    uv run python scripts/make_tables.py --check  # exit 1 if README is stale
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from aerosentinel.utils import get_paths

Result = dict[str, Any]

CMAPSS_MODELS: dict[str, str] = {
    "constant": "Constant",
    "ridge": "Ridge",
    "lgbm": "LightGBM",
    "cnn1d": "1D-CNN",
    "lstm": "LSTM",
}


def load_results(results_dir: Path) -> list[Result]:
    """Load every results JSON, sorted by file name."""
    return [json.loads(p.read_text()) for p in sorted(results_dir.glob("*.json"))]


def fmt(stat: dict[str, float], digits: int) -> str:
    """Format ``{"mean", "std"}`` as ``mean ± std``."""
    return f"{stat['mean']:.{digits}f} ± {stat['std']:.{digits}f}"


def render_cmapss_table(results: list[Result]) -> str:
    """C-MAPSS FD001–FD004 × model table (mean ± std over seeds) plus a footnote.

    Parameters
    ----------
    results : list of dict
        Results documents; non-C-MAPSS ones are ignored.

    Returns
    -------
    str
        Markdown (table + notes), without markers.
    """
    rows = [
        r for r in results if r["dataset"].startswith("cmapss-") and r["model"] in CMAPSS_MODELS
    ]
    if not rows:
        return "_No C-MAPSS results yet. Run `uv run python scripts/run_cmapss_baselines.py`._"
    order = list(CMAPSS_MODELS)
    rows.sort(key=lambda r: (r["dataset"], order.index(r["model"]), r["setting"]))
    best: dict[str, float] = {}
    for r in rows:
        m = r["summary"]["test_capped"]["rmse"]["mean"]
        best[r["dataset"]] = min(best.get(r["dataset"], float("inf")), m)

    head = (
        "| Subset | Model | RMSE (capped) | Score Σ (capped) | MAE (capped) "
        "| RMSE (uncapped) | Score Σ (uncapped) |\n"
        "|---|---|---:|---:|---:|---:|---:|"
    )
    lines = [head]
    for r in rows:
        c, u = r["summary"]["test_capped"], r["summary"]["test_uncapped"]
        rmse_c = fmt(c["rmse"], 2)
        if c["rmse"]["mean"] == best[r["dataset"]]:
            rmse_c = f"**{rmse_c}**"
        subset = r["dataset"].removeprefix("cmapss-").upper()
        lines.append(
            f"| {subset} | {CMAPSS_MODELS[r['model']]} | {rmse_c} | {fmt(c['score_sum'], 0)} "
            f"| {fmt(c['mae'], 2)} | {fmt(u['rmse'], 2)} | {fmt(u['score_sum'], 0)} |"
        )

    seeds = sorted({tuple(r["seeds"]) for r in rows})
    by_subset: dict[str, Result] = {}
    for r in rows:
        by_subset.setdefault(r["dataset"], r)
    units = ", ".join(
        f"{d.removeprefix('cmapss-').upper()} {r['n_train_units']}/{r['n_val_units']}/"
        f"{r['n_test_units']} ({r['n_padded_test_units']} padded)"
        for d, r in sorted(by_subset.items())
    )
    alarms = [r["experiment"] for r in rows if r["leakage_alarm"]["triggered"]]
    notes = [
        "",
        f"Mean ± sample std over seeds {', '.join(map(str, seeds[0]))} "
        f"({'all experiments' if len(seeds) == 1 else 'see JSONs'}). "
        "Lower is better everywhere; **bold** = best capped RMSE per subset. "
        "Score Σ = summed Saxena score with d = pred − true (late errors cost more).",
        "",
        "Protocol: unit-level split (val = 20% of train-file units, fixed split seed); regime "
        "normalization and sensor selection (variance threshold, plus sensors with ≤ 2 distinct "
        "raw values in every regime) fit on train-split units only; "
        "window 30, stride 1; one prediction per test unit from its last window; train targets "
        "capped at 125. Units train/val/test: " + units + ". Test units shorter than the window "
        "are edge-padded by repeating their first cycle.",
        "",
        "Constant and Ridge are deterministic, so their std over seeds is 0 by construction.",
    ]
    if alarms:
        notes += ["", f"⚠️ Leakage alarm (capped FD001 RMSE < 11) fired for: {', '.join(alarms)}."]
    return "\n".join([*lines, *notes])


#: Phase 1 targets on capped test truth (SPEC §3).
CMAPSS_TARGETS: dict[str, float] = {"cmapss-fd001": 13.0, "cmapss-fd004": 20.0}


def render_cmapss_targets(results: list[Result]) -> str:
    """Check the SPEC §3 Phase 1 targets.

    Per subset the model is picked by **validation** RMSE (mean over seeds, validation
    units only); the row then reports that model's capped test RMSE. Test numbers never
    take part in the choice.
    """
    lines = [
        "| Target (capped test RMSE) | Model (picked on val) | Val RMSE | Test RMSE (capped) "
        "| Met? |",
        "|---|---|---:|---:|---|",
    ]
    for dataset, target in CMAPSS_TARGETS.items():
        rows = [r for r in results if r["dataset"] == dataset and r["model"] in CMAPSS_MODELS]
        label = f"{dataset.removeprefix('cmapss-').upper()} ≤ {target:g}"
        if not rows:
            lines.append(f"| {label} | – | – | – | not run |")
            continue
        pick = min(rows, key=lambda r: r["summary"]["val"]["rmse"]["mean"])
        test = pick["summary"]["test_capped"]["rmse"]
        met = "yes" if test["mean"] <= target else "**no**"
        lines.append(
            f"| {label} | {CMAPSS_MODELS[pick['model']]} | "
            f"{fmt(pick['summary']['val']['rmse'], 2)} | {fmt(test, 2)} | {met} |"
        )
    return "\n".join(lines)


TABLES: dict[str, Callable[[list[Result]], str]] = {
    "cmapss": render_cmapss_table,
    "cmapss-targets": render_cmapss_targets,
}


def replace_between_markers(text: str, key: str, body: str) -> str:
    """Replace the content between the ``key`` markers (markers kept, idempotent).

    Raises
    ------
    KeyError
        If the README has no (or more than one) marker pair for ``key``.
    """
    start, end = f"<!-- results:{key}:start -->", f"<!-- results:{key}:end -->"
    pat = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
    if len(pat.findall(text)) != 1:
        raise KeyError(f"expected exactly one {start} ... {end} block")
    return pat.sub(lambda _: f"{start}\n{body.strip()}\n{end}", text)


def render_readme(readme: str, results: list[Result]) -> str:
    """Return the README text with every registered table rebuilt."""
    for key, build in TABLES.items():
        readme = replace_between_markers(readme, key, build(results))
    return readme


def main(argv: list[str] | None = None) -> int:
    """Rewrite (or ``--check``) README tables."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="fail if README is out of date")
    args = ap.parse_args(argv)
    paths = get_paths()
    readme_path = paths.root / "README.md"
    old = readme_path.read_text()
    new = render_readme(old, load_results(paths.results))
    if args.check:
        if new != old:
            print("README tables are stale: run scripts/make_tables.py", file=sys.stderr)
            return 1
        print("README tables are up to date.")
        return 0
    readme_path.write_text(new)
    print(f"Updated {len(TABLES)} table block(s) in {readme_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
