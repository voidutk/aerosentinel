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
    units only); the row then reports that model's capped test RMSE, per seed and as
    mean ± std. Test numbers never take part in the choice.

    A target is ``yes`` only if every seed meets it; ``inconclusive`` if the mean meets it
    but some seed does not; ``no`` if the mean misses.
    """
    lines = [
        "| Target (capped test RMSE) | Model (picked on val) | Val RMSE | Test RMSE (capped) "
        "| Test RMSE per seed | Met? |",
        "|---|---|---:|---:|---:|---|",
    ]
    for dataset, target in CMAPSS_TARGETS.items():
        rows = [r for r in results if r["dataset"] == dataset and r["model"] in CMAPSS_MODELS]
        label = f"{dataset.removeprefix('cmapss-').upper()} ≤ {target:g}"
        if not rows:
            lines.append(f"| {label} | – | – | – | – | not run |")
            continue
        pick = min(rows, key=lambda r: r["summary"]["val"]["rmse"]["mean"])
        test = pick["summary"]["test_capped"]["rmse"]
        per_seed = [float(s["test_capped"]["rmse"]) for s in pick.get("per_seed", [])]
        n_ok = sum(v <= target for v in per_seed)
        if test["mean"] > target:
            met = "**no**"
        elif n_ok == len(per_seed):
            met = "yes"
        else:
            met = f"**inconclusive** (mean meets it; {n_ok}/{len(per_seed)} seeds do)"
        seeds = ", ".join(f"{v:.2f}" for v in per_seed) or "–"
        lines.append(
            f"| {label} | {CMAPSS_MODELS[pick['model']]} | "
            f"{fmt(pick['summary']['val']['rmse'], 2)} | {fmt(test, 2)} | {seeds} | {met} |"
        )
    return "\n".join(lines)


# --- N-CMAPSS (Phase 2) --------------------------------------------------------------------

NCMAPSS_SETTING = {"full": "full flight", "cruise": "cruise only"}


def _ncmapss_runs(results: list[Result]) -> list[Result]:
    rows = [r for r in results if r["dataset"].startswith("ncmapss-") and r["model"] != "data"]
    order = {"full": 0, "cruise": 1}

    def key(r: Result) -> tuple[str, str, int, int]:
        parts = r["setting"].split("-")
        return (r["dataset"], r["model"], order.get(parts[1], 9), 0 if "res" in parts else 1)

    return sorted(rows, key=key)


def _inputs(r: Result) -> str:
    return "W + X_s + residuals" if "residuals" in r["protocol"]["inputs"] else "W + X_s"


def render_ncmapss_table(results: list[Result]) -> str:
    """N-CMAPSS per-cycle results (primary) plus secondary columns, and a per-unit table."""
    rows = _ncmapss_runs(results)
    if not rows:
        return "_No N-CMAPSS results yet. Run `uv run python scripts/run_ncmapss_baselines.py`._"
    best: dict[str, float] = {}
    for r in rows:
        m = r["summary"]["test_per_cycle_capped"]["rmse"]["mean"]
        best[r["dataset"]] = min(best.get(r["dataset"], float("inf")), m)
    lines = [
        "| Subset | Model | Setting | Inputs | Val RMSE (OOF) | **Test RMSE** | Test score "
        "(mean) | Test RMSE, cycles ≥ 6 | Test RMSE per window | Test RMSE uncapped "
        "| Test cycles scored / without window |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        s = r["summary"]
        rm = fmt(s["test_per_cycle_capped"]["rmse"], 2)
        if s["test_per_cycle_capped"]["rmse"]["mean"] == best[r["dataset"]]:
            rm = f"**{rm}**"
        ex = r["excluded_test_cycles"]["total"]
        lines.append(
            f"| {r['dataset'].removeprefix('ncmapss-').upper()} | LightGBM "
            f"| {NCMAPSS_SETTING[r['setting'].split('-')[1]]} | {_inputs(r)} "
            f"| {fmt(s['oof_val_per_cycle_capped']['rmse'], 2)} | {rm} "
            f"| {fmt(s['test_per_cycle_capped']['score_mean'], 2)} "
            f"| {fmt(s['test_per_cycle_ge6_capped']['rmse'], 2)} "
            f"| {fmt(s['test_per_window_capped']['rmse'], 2)} "
            f"| {fmt(s['test_per_cycle_uncapped']['rmse'], 2)} "
            f"| {ex['n_scored']} / {ex['n_without_window']} |"
        )
    units = rows[0]["test_units"]
    lines += [
        "",
        "Per test unit (per-cycle RMSE, capped truth):",
        "",
        "| Setting | Inputs | " + " | ".join(f"unit {u}" for u in units) + " |",
        "|---|---|" + "---:|" * len(units),
    ]
    for r in rows:
        pu = r["summary"]["test_per_unit_per_cycle_capped"]
        cells = " | ".join(fmt(pu[str(u)]["rmse"], 2) if pu[str(u)] else "–" for u in units)
        lines.append(f"| {NCMAPSS_SETTING[r['setting'].split('-')[1]]} | {_inputs(r)} | {cells} |")
    r0 = rows[0]
    cfg = r0["config"]["data"]
    lines += [
        "",
        f"Mean ± sample std over seeds {', '.join(map(str, r0['seeds']))}. Per-cycle metrics "
        "(median of a cycle's window predictions) are primary; RMSE in cycles, lower is better; "
        "**bold** = best test RMSE. Score = mean Saxena score (d = pred − true). "
        f"Truth capped at {cfg['cap']} unless marked uncapped. Windows: T = {cfg['window']} "
        f"model samples at downsample {cfg['downsample_factor']} (one sample every "
        f"{cfg['downsample_factor']} s), stride {cfg['stride']} for every split. "
        "Val = out-of-fold per-cycle RMSE on the calibration unit of each of the 6 rotation "
        "folds (4 train / 1 early-stopping / 1 calib unit); the test prediction is the mean of "
        "the 6 fold models. Residual models are trained and scored from cycle 6 (their "
        'baseline uses cycles 1–5), so compare runs on the "cycles ≥ 6" column.',
    ]
    return "\n".join(lines)


def _find(results: list[Result], experiment: str) -> Result | None:
    return next((r for r in results if r["experiment"] == experiment), None)


def render_ncmapss_data(results: list[Result]) -> str:
    """Phase 2 data audits: Fc audit, caps, cruise mask, window coverage, extrapolation."""
    out: list[str] = []
    fc = _find(results, "ncmapss-ds02_data_fc-audit")
    if fc:
        per = fc["audit"]
        out.append(
            "**Flight classes (dev units).** "
            + "; ".join(
                f"{ds}: "
                + ", ".join(f"class {c}: {len(u)}" for c, u in a["dev_units_per_class"].items())
                for ds, a in per.items()
            )
            + f". Leave-one-flight-class-out will run on **{fc['lofco_dataset']}** "
            "(rule: every class needs ≥ 2 dev units)."
        )
    cap = _find(results, "ncmapss-pooled_data_cap-check")
    if cap:
        cb = cap.get("cruisebench", {})
        eta = cb.get("caps_by_eta", {})
        out += [
            "",
            "**RUL caps.** "
            + (
                f"CruiseBench commit `{cb['commit'][:10]}`. "
                if cb.get("reachable")
                else "CruiseBench not reachable. "
            )
            + f"Caps used: **{cap['cap_source']}** — {cap.get('cap_source_reason', '')}.",
            "",
            "| Subset | SPEC (η = 5 %) | CruiseBench η = 5 % | CruiseBench η = 0 % "
            "| hs-derived (dev min abnormal cycles) | Used |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        used = cap.get("caps_used") or {}
        for ds in cap["hs_derived"]["caps"]:
            out.append(
                f"| {ds} | {cap['spec_caps_eta5'][ds]} "
                f"| {eta.get('eta5', {}).get('caps', {}).get(ds, '–')} "
                f"| {eta.get('eta0', {}).get('caps', {}).get(ds, '–')} "
                f"| {cap['hs_derived']['caps'][ds]} | {used.get(ds, '–')} |"
            )
        mc = cap.get("mask_comparison")
        if mc:
            iou = mc["iou"]
            out += [
                "",
                f"**Cruise mask vs CruiseBench** ({mc['dataset']}, {mc['n_flights_compared']} "
                f"flights): per-flight IoU median {iou['median']:.3f}, mean {iou['mean']:.3f}, "
                f"10th percentile {iou['p10']:.3f}, min {iou['min']:.3f}; "
                f"{iou['share_ge_0.9']:.0%} of flights ≥ 0.9. Flights with a cruise run: "
                f"ours {mc['accepted_ours']}, CruiseBench {mc['accepted_cruisebench']}. "
                f"Mean run length where both accept: ours {mc['mean_run_length_ours'] / 60:.0f} "
                f"min, CruiseBench {mc['mean_run_length_cruisebench'] / 60:.0f} min.",
            ]
    cov = _find(results, "ncmapss-ds02_data_window-coverage")
    if cov:
        grid = [g for g in cov["grid"] if g["min_cycle"] == 1]
        ts = sorted({g["window"] for g in grid})
        out += [
            "",
            "**Cycles with ≥ 1 window** (DS02 dev, all cycles). "
            f"Rule: {cov['rule']} → **T = {cov['chosen_window_ds10']}**.",
            "",
            "| Setting | Downsample | " + " | ".join(f"T = {t}" for t in ts) + " |",
            "|---|---:|" + "---:|" * len(ts),
        ]
        for setting in ("full_flight", "cruise_only"):
            for k in sorted({g["downsample"] for g in grid}):
                cells = []
                for t in ts:
                    g = next(
                        x
                        for x in grid
                        if x["setting"] == setting and x["downsample"] == k and x["window"] == t
                    )
                    cells.append(f"{g['fraction']:.1%}")
                out.append(f"| {setting} | {k} | " + " | ".join(cells) + " |")
    br = _find(results, "ncmapss-ds02_data_baseline-residuals")
    if br:
        parts = []
        for split in ("dev", "test"):
            us = br["splits"][split]["units"]
            parts.append(
                f"{split}: "
                + ", ".join(f"{u} {v['w_extrapolation_share']:.1%}" for u, v in us.items())
            )
        out += [
            "",
            "**W-extrapolation share** (rows after cycle 5 whose flight descriptors leave the "
            "range seen in the engine's own cycles 1–5; the risk of the healthy baseline): "
            + "; ".join(parts)
            + ".",
        ]
    return "\n".join(out) if out else "_No N-CMAPSS data audits yet._"


TABLES: dict[str, Callable[[list[Result]], str]] = {
    "cmapss": render_cmapss_table,
    "cmapss-targets": render_cmapss_targets,
    "ncmapss": render_ncmapss_table,
    "ncmapss-data": render_ncmapss_data,
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
