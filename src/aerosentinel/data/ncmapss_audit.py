"""Dev-unit audits read straight from the N-CMAPSS ``A_dev`` / ``Y_dev`` arrays.

Used by ``scripts/audit_flight_classes.py`` (SPEC §5.5) and ``scripts/check_caps.py``
(SPEC §1.3 item 2). Only dev units are read: official test units never inform the protocol.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numpy.typing import NDArray

from aerosentinel.data.ncmapss import decode_names


def read_dev_meta(h5_path: Path, chunk_rows: int = 2_000_000) -> dict[str, NDArray[Any]]:
    """``unit``, ``cycle``, ``Fc``, ``hs`` and ``Y`` of every dev row (chunked reads)."""
    with h5py.File(h5_path, "r") as f:
        names = decode_names(f["A_var"][()])
        n = int(f["A_dev"].shape[0])
        cols = {k: np.empty(n, dtype=np.float64) for k in ("unit", "cycle", "Fc", "hs", "Y")}
        for a in range(0, n, chunk_rows):
            blk = f["A_dev"][a : a + chunk_rows]
            for k in ("unit", "cycle", "Fc", "hs"):
                cols[k][a : a + len(blk)] = blk[:, names.index(k)]
            cols["Y"][a : a + len(blk)] = f["Y_dev"][a : a + chunk_rows, 0]
    return cols


def per_unit_summary(meta: dict[str, NDArray[Any]]) -> dict[int, dict[str, Any]]:
    """Per dev unit: flight class(es), life, hs onset cycle and abnormal-degradation length.

    The abnormal-degradation length is the number of cycles with ``hs == 0`` (the cycles
    from the onset to the end of life inclusive), which equals ``Y`` at the onset cycle + 1
    because ``Y`` reaches 0 on the last cycle.
    """
    out: dict[int, dict[str, Any]] = {}
    unit = meta["unit"].astype(np.int64)
    for u in np.unique(unit):
        m = unit == u
        cyc = meta["cycle"][m].astype(np.int64)
        hs = meta["hs"][m]
        y = meta["Y"][m].astype(np.int64)
        fcs = sorted(int(v) for v in np.unique(meta["Fc"][m]))
        abn = np.unique(cyc[hs == 0])
        onset = int(abn.min()) if abn.size else None
        out[int(u)] = {
            "Fc": fcs,
            "n_cycles": int(np.unique(cyc).size),
            "last_cycle": int(cyc.max()),
            "rul_at_first_cycle": int(y[0]),
            "rul_at_last_cycle": int(y[-1]),
            "hs_onset_cycle": onset,
            "rul_at_onset": int(y[cyc == onset][0]) if onset is not None else None,
            "n_abnormal_cycles": int(abn.size),
            "hs_monotone": bool(onset is None or np.all(hs[cyc >= onset] == 0)),
        }
    return out
