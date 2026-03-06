"""Tests for stattools.commands.func_cmd."""

import argparse
import io as _io
import os
import sys
import tempfile

import pandas as pd
import pytest

from stattools.commands.func_cmd import FuncCommand


@pytest.fixture
def df():
    return pd.DataFrame({
        "group": ["A", "A", "B", "B", "B"],
        "value": [1.0, 3.0, 2.0, 4.0, 6.0],
        "label": ["x", "y", "x", "y", "z"],
    })


def _run(df, **kwargs):
    with tempfile.NamedTemporaryFile(suffix=".tsv", mode="w", delete=False) as f:
        df.to_csv(f, sep="\t", index=False)
        fname = f.name
    defaults = dict(
        DATAFILE=fname,
        groupcol=None, destcol=None,
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
        FuncCommand().execute(args)
    finally:
        sys.stdout = old
        os.unlink(fname)
    lines = [l for l in buf.getvalue().splitlines() if l.strip()]
    header = lines[0].split("\t")
    rows = [dict(zip(header, l.split("\t"))) for l in lines[1:]]
    return rows


# ---------------------------------------------------------------------------
# cumsum
# ---------------------------------------------------------------------------

class TestCumsum:

    def test_cumsum_produces_column(self, df):
        rows = _run(df, col="value", transform="cumsum")
        assert "value_cumsum" in rows[0]

    def test_cumsum_values(self, df):
        rows = _run(df, col="value", transform="cumsum")
        expected = [1.0, 4.0, 6.0, 10.0, 16.0]
        actual = [float(r["value_cumsum"]) for r in rows]
        assert actual == pytest.approx(expected)

    def test_cumsum_grouped(self, df):
        rows = _run(df, col="value", transform="cumsum", groupcol=["group"])
        # A: 1, 4  B: 2, 6, 12
        expected = {"A": [1.0, 4.0], "B": [2.0, 6.0, 12.0]}
        by_group: dict[str, list[float]] = {}
        for r in rows:
            by_group.setdefault(r["group"], []).append(float(r["value_cumsum"]))
        assert by_group == pytest.approx(expected)

    def test_cumsum_custom_destcol(self, df):
        rows = _run(df, col="value", transform="cumsum", destcol="running_total")
        assert "running_total" in rows[0]

    def test_cumsum_row_count_preserved(self, df):
        rows = _run(df, col="value", transform="cumsum")
        assert len(rows) == len(df)


# ---------------------------------------------------------------------------
# Group aggregates (mean, sum, min, max, count, median, std)
# ---------------------------------------------------------------------------

class TestGroupAggregates:

    def test_group_mean(self, df):
        rows = _run(df, col="value", transform="mean", groupcol=["group"])
        # A mean = 2.0, B mean = 4.0
        a_rows = [float(r["value_mean"]) for r in rows if r["group"] == "A"]
        b_rows = [float(r["value_mean"]) for r in rows if r["group"] == "B"]
        assert all(v == pytest.approx(2.0) for v in a_rows)
        assert all(v == pytest.approx(4.0) for v in b_rows)

    def test_group_sum(self, df):
        rows = _run(df, col="value", transform="sum", groupcol=["group"])
        a_rows = [float(r["value_sum"]) for r in rows if r["group"] == "A"]
        assert all(v == pytest.approx(4.0) for v in a_rows)

    def test_group_min(self, df):
        rows = _run(df, col="value", transform="min", groupcol=["group"])
        a_rows = [float(r["value_min"]) for r in rows if r["group"] == "A"]
        assert all(v == pytest.approx(1.0) for v in a_rows)

    def test_group_max(self, df):
        rows = _run(df, col="value", transform="max", groupcol=["group"])
        b_rows = [float(r["value_max"]) for r in rows if r["group"] == "B"]
        assert all(v == pytest.approx(6.0) for v in b_rows)

    def test_group_count(self, df):
        rows = _run(df, col="value", transform="count", groupcol=["group"])
        a_rows = [float(r["value_count"]) for r in rows if r["group"] == "A"]
        assert all(v == pytest.approx(2.0) for v in a_rows)

    def test_global_mean_no_group(self, df):
        rows = _run(df, col="value", transform="mean")
        # global mean of [1, 3, 2, 4, 6] = 3.2
        means = [float(r["value_mean"]) for r in rows]
        assert all(m == pytest.approx(3.2) for m in means)

    def test_row_count_preserved(self, df):
        rows = _run(df, col="value", transform="mean", groupcol=["group"])
        assert len(rows) == len(df)


# ---------------------------------------------------------------------------
# rank / pct_rank
# ---------------------------------------------------------------------------

class TestRank:

    def test_rank_produces_column(self, df):
        rows = _run(df, col="value", transform="rank")
        assert "value_rank" in rows[0]

    def test_rank_grouped(self, df):
        rows = _run(df, col="value", transform="rank", groupcol=["group"])
        # A: values 1,3 → ranks 1,2  B: values 2,4,6 → ranks 1,2,3
        a_ranks = [float(r["value_rank"]) for r in rows if r["group"] == "A"]
        b_ranks = [float(r["value_rank"]) for r in rows if r["group"] == "B"]
        assert sorted(a_ranks) == pytest.approx([1.0, 2.0])
        assert sorted(b_ranks) == pytest.approx([1.0, 2.0, 3.0])

    def test_pct_rank_range(self, df):
        rows = _run(df, col="value", transform="pct_rank")
        pcts = [float(r["value_pct_rank"]) for r in rows]
        assert all(0.0 < p <= 1.0 for p in pcts)


# ---------------------------------------------------------------------------
# qcut
# ---------------------------------------------------------------------------

class TestQcut:

    def test_qcut_produces_column(self, df):
        rows = _run(df, col="value", transform="qcut:2")
        assert "value_qcut_2" in rows[0]

    def test_qcut_bins_in_range(self, df):
        rows = _run(df, col="value", transform="qcut:2")
        bins = {int(r["value_qcut_2"]) for r in rows}
        assert bins.issubset({1, 2})

    def test_qcut_grouped(self, df):
        rows = _run(df, col="value", transform="qcut:2", groupcol=["group"])
        # Each group should only have bins 1 or 2
        bins = {int(r["value_qcut_2"]) for r in rows}
        assert bins.issubset({1, 2})

    def test_qcut_invalid_n(self, df):
        import tempfile, os as _os
        with tempfile.NamedTemporaryFile(suffix=".tsv", mode="w", delete=False) as f:
            df.to_csv(f, sep="\t", index=False)
            fname = f.name
        try:
            args = argparse.Namespace(
                DATAFILE=fname, col="value", transform="qcut:1", groupcol=None,
                destcol=None, select=None, drop=None, move=None, na_rep=None,
                dropna=False, postquery=[], cast=None, sortasc=None, sortdesc=None,
                sort=None, expect=None, round=None, sigdig=None,
                movetofront=None, movetoback=None, deduplicate=None,
                noheader=False, removeheader=False, output=None, digits=None,
                errortag="-", backend="pandas", nrows=None, delimiter=None,
                readasobject=None, prequery=[],
            )
            with pytest.raises(ValueError, match="qcut requires N >= 2"):
                FuncCommand().execute(args)
        finally:
            _os.unlink(fname)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:

    def test_missing_col_raises(self, df):
        import tempfile, os as _os
        with tempfile.NamedTemporaryFile(suffix=".tsv", mode="w", delete=False) as f:
            df.to_csv(f, sep="\t", index=False)
            fname = f.name
        try:
            args = argparse.Namespace(
                DATAFILE=fname, col="no_such_col", transform="cumsum",
                groupcol=None, destcol=None,
                select=None, drop=None, move=None, na_rep=None, dropna=False,
                postquery=[], cast=None, sortasc=None, sortdesc=None, sort=None,
                expect=None, round=None, sigdig=None,
                movetofront=None, movetoback=None, deduplicate=None,
                noheader=False, removeheader=False, output=None, digits=None,
                errortag="-", backend="pandas", nrows=None, delimiter=None,
                readasobject=None, prequery=[],
            )
            with pytest.raises(ValueError, match="no_such_col"):
                FuncCommand().execute(args)
        finally:
            _os.unlink(fname)

    def test_unknown_transform_raises(self, df):
        import tempfile, os as _os
        with tempfile.NamedTemporaryFile(suffix=".tsv", mode="w", delete=False) as f:
            df.to_csv(f, sep="\t", index=False)
            fname = f.name
        try:
            args = argparse.Namespace(
                DATAFILE=fname, col="value", transform="bogus",
                groupcol=None, destcol=None,
                select=None, drop=None, move=None, na_rep=None, dropna=False,
                postquery=[], cast=None, sortasc=None, sortdesc=None, sort=None,
                expect=None, round=None, sigdig=None,
                movetofront=None, movetoback=None, deduplicate=None,
                noheader=False, removeheader=False, output=None, digits=None,
                errortag="-", backend="pandas", nrows=None, delimiter=None,
                readasobject=None, prequery=[],
            )
            with pytest.raises(ValueError, match="Unknown transform"):
                FuncCommand().execute(args)
        finally:
            _os.unlink(fname)

    def test_default_destcol_name(self, df):
        rows = _run(df, col="value", transform="cumsum")
        assert "value_cumsum" in rows[0]

    def test_default_destcol_qcut(self, df):
        rows = _run(df, col="value", transform="qcut:4")
        assert "value_qcut_4" in rows[0]
