"""Cap check (SPEC §1.3 item 2) and cruise-mask comparison with CruiseBench (§1.3 item 4).

Usage::

    git clone --depth 1 https://github.com/NostalgiaJohn/CruiseBench <dir>   # read only
    uv run python scripts/check_caps.py --cruisebench <dir>

Writes ``reports/results/ncmapss-pooled_data_cap-check.json`` and
``reports/figures/ncmapss_mask_vs_cruisebench.png``. The CruiseBench checkout is read as
text/HDF5 data only; none of its code is imported or executed.

Decision rules (agreed 2026-11-27):

1. CruiseBench's η=5% caps differ from SPEC, or exceed the hs-derived minimum
   abnormal-degradation length of the dev units  ->  STOP (exit code 4).
2. CruiseBench unreachable, or its cap definition not stated  ->  use the hs-derived caps
   (``cap_source: "hs-derived"``) and continue.
3. Otherwise use the SPEC caps (``cap_source: "spec"``).

The mask comparison is informational and never stops the pipeline.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from aerosentinel.data.masks import DEFAULT_MASK, interval_iou
from aerosentinel.data.ncmapss import UnitStore
from aerosentinel.data.ncmapss_audit import per_unit_summary, read_dev_meta
from aerosentinel.data.ncmapss_derive import load_cruise_runs
from aerosentinel.runs import git_info
from aerosentinel.utils import get_paths
from aerosentinel.viz import SERIES, TEXT_2, apply_style, save

DATASETS = ("DS01", "DS02", "DS03", "DS04", "DS05", "DS06", "DS07", "DS08a", "DS08c")
SPEC_CAPS = dict(zip(DATASETS, (38, 34, 34, 43, 38, 41, 39, 34, 28), strict=True))
EXTRACTED = DATASETS[:7]
STOP_EXIT_CODE = 4


def parse_cruisebench_caps(repo: Path) -> dict[str, Any]:
    """Cap arrays per η label from ``scripts/run_baseline_grid.sh`` (with line numbers)."""
    path = repo / "scripts" / "run_baseline_grid.sh"
    lines = path.read_text().splitlines()
    out: dict[str, Any] = {}
    label = None
    for no, line in enumerate(lines, 1):
        if m := re.search(r'ETA_LABEL="(eta\d+)"', line):
            label = m.group(1)
        if (m := re.search(r"MAX_RUL_CAPS=\(([\d\s]+)\)", line)) and label:
            caps = [int(v) for v in m.group(1).split()]
            out[label] = {
                "caps": dict(zip(DATASETS, caps, strict=True)),
                "source": f"scripts/run_baseline_grid.sh:{no}",
            }
    return out


def find_eta_definition(repo: Path) -> list[str]:
    """Lines that could define how η turns into a cap (beyond listing the cap values)."""
    hits = []
    # A definition must tie η to a degradation quantity on the same line; plain cap lists
    # (``MAX_RUL_CAPS=...``, "eta=5% caps | 38, ...") and unrelated hs comments do not count.
    eta = re.compile(r"(\beta\b|η|eta[_ ]?percent)", re.I)
    deg = re.compile(r"(abnormal|onset|\bhs\b|health|degradation|fraction|percent of|share)", re.I)
    for p in sorted(repo.rglob("*")):
        if p.suffix not in {".py", ".md", ".sh", ".txt"} or ".git" in p.parts:
            continue
        for no, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
            if eta.search(line) and deg.search(line) and "MAX_RUL_CAPS" not in line:
                hits.append(f"{p.relative_to(repo)}:{no}: {line.strip()[:160]}")
    return hits


def hs_caps(raw: Path) -> dict[str, Any]:
    """Per extracted subset: dev-unit abnormal-degradation lengths from hs."""
    out: dict[str, Any] = {}
    for ds in EXTRACTED:
        (h5,) = sorted(raw.glob(f"N-CMAPSS_{ds}*.h5"))
        units = per_unit_summary(read_dev_meta(h5))
        lengths = {str(u): s["n_abnormal_cycles"] for u, s in units.items()}
        out[ds] = {
            "source": h5.name,
            "n_dev_units": len(units),
            "per_unit": {str(u): s for u, s in units.items()},
            "min_abnormal_cycles": min(lengths.values()),
            "argmin_unit": int(min(lengths, key=lambda k: lengths[k])),
            "min_rul_at_onset": min(s["rul_at_onset"] for s in units.values()),
            "all_hs_monotone": all(s["hs_monotone"] for s in units.values()),
        }
    return out


def mask_comparison(repo: Path, store: UnitStore) -> dict[str, Any]:
    """Per-flight IoU between our cruise run and CruiseBench's released DS02 dev mask."""
    with h5py.File(repo / "dataset" / "cruise_mask" / "common_altitude_cruise_mask.h5") as f:
        rows = f["cycles"][()]
    theirs: dict[tuple[int, int], tuple[int, int] | None] = {}
    for r in rows:
        if r["dataset"].decode() == "DS02-006" and r["split"].decode() == "dev":
            iv = (int(r["start_sample"]), int(r["end_sample"]) + 1) if r["detected"] else None
            theirs[(int(r["unit"]), int(r["cycle"]))] = iv  # inclusive end -> half-open
    flights = []
    for u in store.units:
        fl = np.load(store.unit_dir(u) / "flights.npy")
        runs = load_cruise_runs(store, u, DEFAULT_MASK)
        for (cyc, s, e), (rs, re_) in zip(fl, runs, strict=True):
            ours = (int(rs - s), int(re_ - s)) if rs >= 0 else None
            key = (int(u), int(cyc))
            if key not in theirs:
                continue
            tb = theirs[key]
            flights.append(
                {
                    "unit": int(u),
                    "cycle": int(cyc),
                    "n_samples": int(e - s),
                    "ours": ours,
                    "cruisebench": tb,
                    "iou": interval_iou(ours, tb),
                }
            )
    iou = np.array([f["iou"] for f in flights])
    both = [f for f in flights if f["ours"] and f["cruisebench"]]
    return {
        "dataset": "DS02 dev",
        "n_flights_compared": len(flights),
        "n_flights_missing_in_cruisebench": sum(
            int(np.load(store.unit_dir(u) / "flights.npy").shape[0]) for u in store.units
        )
        - len(flights),
        "accepted_ours": sum(f["ours"] is not None for f in flights),
        "accepted_cruisebench": sum(f["cruisebench"] is not None for f in flights),
        "accepted_both": len(both),
        "iou": {
            "mean": float(iou.mean()),
            "median": float(np.median(iou)),
            "p10": float(np.quantile(iou, 0.10)),
            "min": float(iou.min()),
            "share_ge_0.9": float((iou >= 0.9).mean()),
        },
        "mean_run_length_ours": float(np.mean([f["ours"][1] - f["ours"][0] for f in both])),
        "mean_run_length_cruisebench": float(
            np.mean([f["cruisebench"][1] - f["cruisebench"][0] for f in both])
        ),
        "worst_flights": sorted(flights, key=lambda f: f["iou"])[:3],
    }


def mask_figure(store: UnitStore, worst: list[dict[str, Any]], out: Path) -> None:
    """Altitude of the 3 lowest-IoU flights with both cruise runs shaded."""
    import matplotlib.pyplot as plt

    apply_style()
    alt_col = store.var_names["W"].index("alt")
    fig, axes = plt.subplots(len(worst), 1, figsize=(8, 2.3 * len(worst)), sharex=False)
    for ax, f in zip(np.atleast_1d(axes), worst, strict=True):
        ua = store.load(f["unit"])
        fl = ua.flights[ua.flights[:, 0] == f["cycle"]][0]
        alt = np.asarray(ua.W[fl[1] : fl[2], alt_col])
        t = np.arange(alt.size) / 60.0
        for iv, color, label in (
            (f["ours"], SERIES[0], "AeroSentinel run"),
            (f["cruisebench"], SERIES[1], "CruiseBench run"),
        ):
            if iv is not None:
                ax.axvspan(iv[0] / 60, iv[1] / 60, color=color, alpha=0.22, lw=0, label=label)
        ax.plot(t, alt, color=TEXT_2, lw=1.0, label="altitude")
        none = [n for n, iv in (("ours", f["ours"]), ("CruiseBench", f["cruisebench"])) if not iv]
        extra = f" ({' and '.join(none)}: no run accepted)" if none else ""
        ax.set_title(
            f"unit {f['unit']}, cycle {f['cycle']} — IoU {f['iou']:.2f}{extra}", loc="left"
        )
        ax.set_ylabel("altitude [ft]")
    np.atleast_1d(axes)[-1].set_xlabel("time in flight [min] (1 Hz samples)")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [
        Patch(color=SERIES[0], alpha=0.22, label="AeroSentinel run"),
        Patch(color=SERIES[1], alpha=0.22, label="CruiseBench run"),
        Line2D([], [], color=TEXT_2, lw=1.0, label="altitude"),
    ]
    np.atleast_1d(axes)[0].legend(handles=handles, loc="upper right", ncol=3)
    fig.suptitle(
        "Cruise run: our mask vs CruiseBench's released mask (3 lowest-IoU DS02 dev flights)",
        x=0.01,
        ha="left",
        fontsize=10,
    )
    fig.tight_layout()
    save(fig, out)


def main(argv: list[str] | None = None) -> int:
    """Run the checks, write the JSON (and figure), return 0 or the stop exit code."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cruisebench", type=Path, required=True, help="CruiseBench checkout")
    ap.add_argument(
        "--also-search",
        type=Path,
        nargs="*",
        default=[],
        help="other CruiseBench branch checkouts searched for the η definition",
    )
    args = ap.parse_args(argv)
    paths = get_paths()
    repo: Path = args.cruisebench
    hs = hs_caps(paths.data_raw / "ncmapss")
    hs_cap = {ds: v["min_abnormal_cycles"] for ds, v in hs.items()}

    doc: dict[str, Any] = {
        "experiment": "ncmapss-pooled_data_cap-check",
        "dataset": "ncmapss-pooled",
        "model": "data",
        "setting": "cap-check",
        "written_by": "scripts/check_caps.py",
        "spec_caps_eta5": SPEC_CAPS,
        "hs_derived": {
            "definition": "min over dev units of the number of cycles with hs == 0 "
            "(abnormal-degradation length; = RUL at the hs onset cycle + 1). Dev units only.",
            "caps": hs_cap,
            "per_dataset": hs,
        },
        "git": git_info(paths.root),
    }
    reachable = (repo / "scripts" / "run_baseline_grid.sh").is_file()
    stop_reasons: list[str] = []
    if not reachable:
        doc["cruisebench"] = {"reachable": False}
        doc["cap_source"] = "hs-derived"
        doc["cap_source_reason"] = f"CruiseBench checkout not found at {repo}"
    else:
        commit = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        cb = parse_cruisebench_caps(repo)
        definition_hits = find_eta_definition(repo)
        searched = [{"path": "main checkout", "commit": commit}]
        for other in args.also_search:
            oc = subprocess.run(
                ["git", "-C", str(other), "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
            ).stdout.strip()
            osha = subprocess.run(
                ["git", "-C", str(other), "rev-parse", "HEAD"], capture_output=True, text=True
            ).stdout.strip()
            searched.append({"branch": oc, "commit": osha})
            definition_hits += [f"[{oc}] {h}" for h in find_eta_definition(other)]
        eta5 = cb.get("eta5", {}).get("caps", {})
        match_spec = {ds: eta5.get(ds) == SPEC_CAPS[ds] for ds in DATASETS}
        within_hs = {ds: eta5.get(ds, 10**9) <= hs_cap[ds] for ds in EXTRACTED}
        eta0 = cb.get("eta0", {}).get("caps", {})
        doc["cruisebench"] = {
            "reachable": True,
            "repo": "https://github.com/NostalgiaJohn/CruiseBench",
            "commit": commit,
            "caps_by_eta": cb,
            "eta5_matches_spec": match_spec,
            "eta5_within_hs_min_abnormal_length": within_hs,
            "eta0_equals_hs_derived_dev_min": {ds: eta0.get(ds) == hs_cap[ds] for ds in EXTRACTED},
            "eta_definition_found": bool(definition_hits),
            "eta_definition_searched": searched,
            "eta_definition_search_hits": definition_hits,
            "note": "The repo lists cap values per η (0, 5, 10 %) but does not state how η "
            "maps to a cap, so the η=5% caps cannot be recomputed from hs.",
        }
        if not all(match_spec.values()):
            stop_reasons.append(f"CruiseBench η=5% caps differ from SPEC: {match_spec}")
        if not all(within_hs.values()):
            stop_reasons.append(f"η=5% cap exceeds the hs-derived dev minimum: {within_hs}")
        if stop_reasons:
            doc["cap_source"] = None
        elif not definition_hits:
            doc["cap_source"] = "hs-derived"
            doc["cap_source_reason"] = (
                "CruiseBench's η=5% caps match SPEC but its cap definition is not stated "
                "(ambiguous), so per the agreed rule the hs-derived dev-unit caps are used"
            )
        else:
            doc["cap_source"] = "spec"
            doc["cap_source_reason"] = "CruiseBench caps match SPEC and its definition is stated"
        store = UnitStore(paths.data_processed / "ncmapss", "DS02", "dev")
        cmp = mask_comparison(repo, store)
        doc["mask_comparison"] = {
            "ours": {"params": DEFAULT_MASK.to_dict(), "module": "aerosentinel.data.masks"},
            "cruisebench": "dataset/cruise_mask/common_altitude_cruise_mask.h5 (released mask; "
            "inclusive cycle-local intervals, converted to half-open)",
            **cmp,
        }
        paths.figures.mkdir(parents=True, exist_ok=True)
        fig_path = paths.figures / "ncmapss_mask_vs_cruisebench.png"
        mask_figure(store, cmp["worst_flights"], fig_path)
        doc["mask_comparison"]["figure"] = fig_path.relative_to(paths.root).as_posix()
    doc["stop"] = bool(stop_reasons)
    doc["stop_reasons"] = stop_reasons
    if doc["cap_source"] == "hs-derived":
        doc["caps_used"] = hs_cap
    elif doc["cap_source"] == "spec":
        doc["caps_used"] = {ds: SPEC_CAPS[ds] for ds in EXTRACTED}
    else:
        doc["caps_used"] = None
    out = paths.results / f"{doc['experiment']}.json"
    paths.results.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2) + "\n")
    print(json.dumps({k: doc[k] for k in ("cap_source", "caps_used", "stop", "stop_reasons")}))
    if "mask_comparison" in doc:
        print("mask IoU:", json.dumps(doc["mask_comparison"]["iou"]))
    print(f"-> {out}")
    return STOP_EXIT_CODE if stop_reasons else 0


if __name__ == "__main__":
    raise SystemExit(main())
