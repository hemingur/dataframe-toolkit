"""Tests for stattools.commands.interp_cmd."""

import argparse
import io as _io
import math
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

from stattools.commands.interp_cmd import InterpCommand


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tsv(df: pd.DataFrame) -> str:
    f = tempfile.NamedTemporaryFile(suffix=".tsv", mode="w", delete=False)
    df.to_csv(f, sep="\t", index=False)
    f.close()
    return f.name


def _run(data_df: pd.DataFrame, ref_df: pd.DataFrame, **kwargs):
    data_file = _tsv(data_df)
    ref_file = _tsv(ref_df)
    defaults = dict(
        DATAFILE=data_file,
        ref=ref_file,
        xcol="x",
        refx=None,
        val=["y"],
        destcol=None,
        groupcol=None,
        method="linear",
        fill="nan",
        select=None, drop=None, move=None, na_rep=None, dropna=False,
        postquery=[], cast=None, sortasc=None, sortdesc=None, sort=None,
        expect=None, round=None, sigdig=None,
        movetofront=None, movetoback=None,
        deduplicate=None, noheader=False, removeheader=False,
        output=None, digits=None, errortag="-",
        backend="pandas", nrows=None, delimiter=None,
        readasobject=None, prequery=[],
    )
    defaults.update(kwargs)
    args = argparse.Namespace(**defaults)
    buf = _io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        InterpCommand().execute(args)
    finally:
        sys.stdout = old
        os.unlink(data_file)
        os.unlink(ref_file)
    lines = [l for l in buf.getvalue().splitlines() if l.strip()]
    header = lines[0].split("\t")
    rows = [dict(zip(header, l.split("\t"))) for l in lines[1:]]
    # Replace empty strings (NaN in TSV) with float("nan") for convenience
    for r in rows:
        for k, v in r.items():
            if v == "":
                r[k] = float("nan")
    return rows


@pytest.fixture
def linear_ref():
    """Reference curve: y = 2*x for x in [0, 10]."""
    x = np.linspace(0, 10, 11)
    return pd.DataFrame({"x": x, "y": 2 * x})


@pytest.fixture
def data_df():
    return pd.DataFrame({"x": [0.0, 2.5, 5.0, 7.5, 10.0], "label": list("abcde")})


# ---------------------------------------------------------------------------
# Basic interpolation
# ---------------------------------------------------------------------------

class TestInterpBasic:

    def test_column_added(self, data_df, linear_ref):
        rows = _run(data_df, linear_ref)
        assert "y" in rows[0]

    def test_row_count_preserved(self, data_df, linear_ref):
        rows = _run(data_df, linear_ref)
        assert len(rows) == len(data_df)

    def test_linear_exact_at_reference_points(self, data_df, linear_ref):
        rows = _run(data_df, linear_ref)
        for r in rows:
            x = float(r["x"])
            y = float(r["y"])
            assert y == pytest.approx(2 * x, abs=1e-9)

    def test_linear_midpoint(self):
        ref = pd.DataFrame({"x": [0.0, 10.0], "y": [0.0, 10.0]})
        data = pd.DataFrame({"x": [5.0]})
        rows = _run(data, ref)
        assert float(rows[0]["y"]) == pytest.approx(5.0)

    def test_custom_destcol(self, data_df, linear_ref):
        rows = _run(data_df, linear_ref, destcol=["interp_y"])
        assert "interp_y" in rows[0]
        assert "y" not in rows[0]

    def test_existing_cols_preserved(self, data_df, linear_ref):
        rows = _run(data_df, linear_ref)
        assert "label" in rows[0]

    def test_different_x_col_names(self, linear_ref):
        # data uses "pos", ref uses "x"
        data = pd.DataFrame({"pos": [0.0, 5.0, 10.0]})
        rows = _run(data, linear_ref, xcol="pos", refx="x")
        assert float(rows[1]["y"]) == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Out-of-range fill
# ---------------------------------------------------------------------------

class TestFill:

    def test_out_of_range_nan_default(self, linear_ref):
        data = pd.DataFrame({"x": [-1.0, 11.0]})
        rows = _run(data, linear_ref, fill="nan")
        for r in rows:
            assert math.isnan(r["y"])

    def test_out_of_range_edge_fill(self, linear_ref):
        data = pd.DataFrame({"x": [-1.0, 11.0]})
        rows = _run(data, linear_ref, fill="edge")
        # Edge values: y(x=-1) → y[0]=0.0, y(x=11) → y[-1]=20.0
        assert float(rows[0]["y"]) == pytest.approx(0.0)
        assert float(rows[1]["y"]) == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# Multiple value columns
# ---------------------------------------------------------------------------

class TestMultipleValCols:

    def test_two_val_cols(self, data_df):
        ref = pd.DataFrame({"x": [0.0, 10.0], "y1": [0.0, 10.0], "y2": [0.0, 20.0]})
        rows = _run(data_df, ref, val=["y1", "y2"], destcol=["out1", "out2"])
        assert "out1" in rows[0]
        assert "out2" in rows[0]
        assert float(rows[2]["out1"]) == pytest.approx(5.0)   # x=5, y1=5
        assert float(rows[2]["out2"]) == pytest.approx(10.0)  # x=5, y2=10

    def test_destcol_mismatch_raises(self, data_df, linear_ref):
        with pytest.raises(ValueError, match="must match"):
            _run(data_df, linear_ref, val=["y"], destcol=["a", "b"])


# ---------------------------------------------------------------------------
# Grouped interpolation
# ---------------------------------------------------------------------------

class TestGrouped:

    def _make_grouped_ref(self):
        """Two groups A and B with different linear curves."""
        df_a = pd.DataFrame({"group": "A", "x": [0.0, 10.0], "y": [0.0, 10.0]})
        df_b = pd.DataFrame({"group": "B", "x": [0.0, 10.0], "y": [0.0, 20.0]})
        return pd.concat([df_a, df_b], ignore_index=True)

    def test_grouped_interpolation(self):
        ref = self._make_grouped_ref()
        data = pd.DataFrame({"group": ["A", "A", "B", "B"], "x": [2.0, 5.0, 2.0, 5.0]})
        rows = _run(data, ref, xcol="x", groupcol=["group"])
        a_rows = {float(r["x"]): float(r["y"]) for r in rows if r["group"] == "A"}
        b_rows = {float(r["x"]): float(r["y"]) for r in rows if r["group"] == "B"}
        assert a_rows[2.0] == pytest.approx(2.0)
        assert a_rows[5.0] == pytest.approx(5.0)
        assert b_rows[2.0] == pytest.approx(4.0)
        assert b_rows[5.0] == pytest.approx(10.0)

    def test_group_missing_in_ref_gives_nan(self):
        ref = pd.DataFrame({"group": ["A"], "x": [0.0], "y": [0.0]})
        # group B has no ref data → NaN
        data = pd.DataFrame({"group": ["B"], "x": [5.0]})
        rows = _run(data, ref, xcol="x", groupcol=["group"])
        assert math.isnan(rows[0]["y"])


# ---------------------------------------------------------------------------
# Interpolation methods
# ---------------------------------------------------------------------------

class TestMethods:

    def test_nearest_method(self):
        ref = pd.DataFrame({"x": [0.0, 10.0], "y": [0.0, 100.0]})
        data = pd.DataFrame({"x": [3.0, 7.0]})
        rows = _run(data, ref, method="nearest")
        # nearest to x=3 → 0, nearest to x=7 → 100
        assert float(rows[0]["y"]) == pytest.approx(0.0)
        assert float(rows[1]["y"]) == pytest.approx(100.0)

    def test_cubic_method(self):
        # Cubic needs at least 4 points
        x = np.linspace(0, 10, 20)
        ref = pd.DataFrame({"x": x, "y": x ** 2})
        data = pd.DataFrame({"x": [5.0]})
        rows = _run(data, ref, method="cubic")
        assert float(rows[0]["y"]) == pytest.approx(25.0, abs=0.1)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:

    def test_missing_xcol_raises(self, linear_ref):
        data = pd.DataFrame({"pos": [1.0, 2.0]})
        with pytest.raises(ValueError, match="no_col"):
            _run(data, linear_ref, xcol="no_col")

    def test_missing_val_col_raises(self, data_df, linear_ref):
        with pytest.raises(ValueError, match="no_val"):
            _run(data_df, linear_ref, val=["no_val"])
