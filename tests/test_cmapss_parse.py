"""C-MAPSS parsing: whitespace splitting, exactly 26 columns, trailing spaces tolerated."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aerosentinel.data.cmapss import COLUMNS, read_cmapss_table, read_rul_file


def _row(unit: int, cycle: int, n_values: int = 24) -> str:
    vals = [f"{0.5 + i:.4f}" for i in range(n_values)]
    return f"{unit} {cycle} " + " ".join(vals)


def test_trailing_and_double_spaces_parse_to_26_columns(tmp_path: Path) -> None:
    p = tmp_path / "train_FD001.txt"
    # Real files end every row with trailing spaces; also inject a doubled space.
    lines = [_row(1, 1) + "  ", _row(1, 2).replace(" ", "  ", 3) + " ", _row(2, 1) + "   "]
    p.write_text("\n".join(lines) + "\n")
    df = read_cmapss_table(p)
    assert list(df.columns) == list(COLUMNS)
    assert df.shape == (3, 26)
    assert df["unit"].dtype == np.int64 and df["cycle"].dtype == np.int64
    assert df["s21"].dtype == np.float64
    assert df.loc[1, "setting1"] == pytest.approx(0.5)
    assert df.loc[2, "s21"] == pytest.approx(23.5)
    assert df[list(COLUMNS)].notna().all().all()


@pytest.mark.parametrize("n_values", [23, 25])
def test_wrong_column_count_raises(tmp_path: Path, n_values: int) -> None:
    p = tmp_path / "bad.txt"
    p.write_text(_row(1, 1, n_values) + " \n" + _row(1, 2, n_values) + " \n")
    with pytest.raises(ValueError, match="expected 26 columns"):
        read_cmapss_table(p)


def test_rul_file_with_trailing_spaces(tmp_path: Path) -> None:
    p = tmp_path / "RUL_FD001.txt"
    p.write_text("112 \n98 \n69 \n")
    np.testing.assert_array_equal(read_rul_file(p), [112, 98, 69])
    p.write_text("1 2\n3 4\n")
    with pytest.raises(ValueError, match="expected 1 column"):
        read_rul_file(p)
