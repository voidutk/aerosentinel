"""Flight-class audit (SPEC §5.5): Fc per DS02 and DS03 dev unit, and the LOFCO dataset.

Usage: ``uv run python scripts/audit_flight_classes.py``
Writes ``reports/results/ncmapss-ds02_data_fc-audit.json``.

Rule: leave-one-flight-class-out runs on DS02 if every flight class (1, 2, 3) has >= 2 DS02
dev units, otherwise on DS03 (if DS03 satisfies the rule; otherwise neither, reported).
"""

from __future__ import annotations

import json

from aerosentinel.data.ncmapss_audit import per_unit_summary, read_dev_meta
from aerosentinel.runs import git_info
from aerosentinel.utils import get_paths

CLASSES = (1, 2, 3)
MIN_UNITS_PER_CLASS = 2


def main() -> int:
    """Audit DS02 and DS03 and write the decision."""
    paths = get_paths()
    raw = paths.data_raw / "ncmapss"
    audit: dict[str, object] = {}
    ok: dict[str, bool] = {}
    for ds in ("DS02", "DS03"):
        (h5,) = sorted(raw.glob(f"N-CMAPSS_{ds}*.h5"))
        units = per_unit_summary(read_dev_meta(h5))
        by_class = {str(c): sorted(u for u, s in units.items() if c in s["Fc"]) for c in CLASSES}
        multi = {str(u): s["Fc"] for u, s in units.items() if len(s["Fc"]) > 1}
        ok[ds] = not multi and all(len(v) >= MIN_UNITS_PER_CLASS for v in by_class.values())
        audit[ds] = {
            "source": h5.name,
            "fc_per_dev_unit": {str(u): s["Fc"] for u, s in units.items()},
            "dev_units_per_class": by_class,
            "units_with_several_classes": multi,
            "rule_satisfied": ok[ds],
        }
        print(f"{ds}: Fc per dev unit {audit[ds]['fc_per_dev_unit']}")  # type: ignore[index]
        print(f"{ds}: dev units per class {by_class} -> rule satisfied: {ok[ds]}")
    chosen = "DS02" if ok["DS02"] else ("DS03" if ok["DS03"] else None)
    doc = {
        "experiment": "ncmapss-ds02_data_fc-audit",
        "dataset": "ncmapss-ds02",
        "model": "data",
        "setting": "fc-audit",
        "written_by": "scripts/audit_flight_classes.py",
        "rule": f"leave-one-flight-class-out on DS02 if every class {CLASSES} has >= "
        f"{MIN_UNITS_PER_CLASS} dev units, else DS03 (SPEC §5.5)",
        "flight_classes": {"1": "1-3 h", "2": "3-5 h", "3": "> 5 h"},
        "audit": audit,
        "lofco_dataset": chosen,
        "git": git_info(paths.root),
    }
    paths.results.mkdir(parents=True, exist_ok=True)
    out = paths.results / f"{doc['experiment']}.json"
    out.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"leave-one-flight-class-out dataset: {chosen} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
