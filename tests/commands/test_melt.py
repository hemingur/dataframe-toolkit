"""Tests for stattools.commands.melt_cmd."""

import argparse
import io as _io
import sys

import pandas as pd
import pytest

from stattools.commands.melt_cmd import MeltCommand


@pytest.fixture
def wide_df():
    return pd.DataFrame(
        {
            "sample": ["s1", "s2"],
            "gene_A": [1.0, 2.0],
            "gene_B": [3.0, 4.0],
            "gene_C": [5.0, 6.0],
        }
    )


def _run(df, **kwargs):
    import os
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".tsv", mode="w", delete=False) as f:
        df.to_csv(f, sep="\t", index=False)
        fname = f.name
    defaults = dict(
        DATAFILE=fname,
        indexcols=None,
        destcol="variable",
        valuecol="value",
        select=None,
        drop=None,
        move=None,
        na_rep=None,
        dropna=False,
        postquery=[],
        cast=None,
        sortasc=None,
        sortdesc=None,
        sort=None,
        round=None,
        deduplicate=None,
        noheader=False,
        removeheader=False,
        output=None,
        digits=None,
        errortag="-",
        backend="pandas",
        nrows=None,
        delimiter=None,
        readasobject=None,
        prequery=[],
    )
    defaults.update(kwargs)
    args = argparse.Namespace(**defaults)
    buf = _io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        MeltCommand().execute(args)
    finally:
        sys.stdout = old
        os.unlink(fname)
    lines = [line for line in buf.getvalue().splitlines() if line.strip()]
    header = lines[0].split("\t")
    rows = [dict(zip(header, line.split("\t"), strict=False)) for line in lines[1:]]
    return rows


class TestMeltBasic:
    def test_all_cols_melted_when_no_index(self, wide_df):
        rows = _run(wide_df)
        assert len(rows) == 8  # 2 samples × 4 cols

    def test_with_index_col(self, wide_df):
        rows = _run(wide_df, indexcols=["sample"])
        assert len(rows) == 6  # 2 samples × 3 value cols

    def test_index_col_preserved(self, wide_df):
        rows = _run(wide_df, indexcols=["sample"])
        assert all("sample" in r for r in rows)

    def test_default_column_names(self, wide_df):
        rows = _run(wide_df, indexcols=["sample"])
        assert "variable" in rows[0]
        assert "value" in rows[0]

    def test_custom_destcol(self, wide_df):
        rows = _run(wide_df, indexcols=["sample"], destcol="gene")
        assert "gene" in rows[0]

    def test_custom_valuecol(self, wide_df):
        rows = _run(wide_df, indexcols=["sample"], valuecol="expression")
        assert "expression" in rows[0]

    def test_variable_values_are_column_names(self, wide_df):
        rows = _run(wide_df, indexcols=["sample"])
        variables = {r["variable"] for r in rows}
        assert variables == {"gene_A", "gene_B", "gene_C"}

    def test_values_are_correct(self, wide_df):
        rows = _run(wide_df, indexcols=["sample"])
        s1_a = next(
            r for r in rows if r["sample"] == "s1" and r["variable"] == "gene_A"
        )
        assert float(s1_a["value"]) == pytest.approx(1.0)


class TestMeltValidation:
    def test_missing_indexcol_raises(self, wide_df):
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".tsv", mode="w", delete=False) as f:
            wide_df.to_csv(f, sep="\t", index=False)
            fname = f.name
        try:
            args = argparse.Namespace(
                DATAFILE=fname,
                indexcols=["no_such_col"],
                destcol="variable",
                valuecol="value",
                select=None,
                drop=None,
                move=None,
                na_rep=None,
                dropna=False,
                postquery=[],
                cast=None,
                sortasc=None,
                sortdesc=None,
                sort=None,
                round=None,
                deduplicate=None,
                noheader=False,
                removeheader=False,
                output=None,
                digits=None,
                errortag="-",
                backend="pandas",
                nrows=None,
                delimiter=None,
                readasobject=None,
                prequery=[],
            )
            with pytest.raises(ValueError, match="no_such_col"):
                MeltCommand().execute(args)
        finally:
            os.unlink(fname)
