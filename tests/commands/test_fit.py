"""
Tests for dftk.commands.fit_cmd.
"""

import argparse
import io as _io
import sys

import numpy as np
import pandas as pd
import pytest

from dftk.commands.fit_cmd import (
    FitCommand,
    regress_it,
    res2df,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_df():
    """y = 2*x + 1 + noise (small, deterministic)."""
    rng = np.random.default_rng(42)
    x = np.arange(20, dtype=float)
    y = 2.0 * x + 1.0 + rng.normal(0, 0.5, 20)
    return pd.DataFrame({"x": x, "y": y})


@pytest.fixture
def grouped_df():
    """Two groups, each with y ~ x relationship."""
    rng = np.random.default_rng(0)
    n = 15
    x = np.arange(n, dtype=float)
    df_a = pd.DataFrame({"group": "A", "x": x, "y": 3.0 * x + rng.normal(0, 0.5, n)})
    df_b = pd.DataFrame({"group": "B", "x": x, "y": -x + 10 + rng.normal(0, 0.5, n)})
    return pd.concat([df_a, df_b], ignore_index=True)


@pytest.fixture
def weighted_df():
    rng = np.random.default_rng(7)
    x = np.arange(20, dtype=float)
    y = 2.0 * x + rng.normal(0, 1, 20)
    w = np.ones(20)
    w[:10] = 0.1  # low weight for first half
    return pd.DataFrame({"x": x, "y": y, "w": w})


def _args(**kwargs) -> argparse.Namespace:
    defaults = dict(
        formula="y ~ x",
        weights=None,
        tag="tag",
        robust=False,
        groupcol=None,
        summary=False,
        anova=False,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# regress_it — basic dispatch
# ---------------------------------------------------------------------------


class TestRegressIt:
    def test_ols_returns_result(self, simple_df):
        res = regress_it(simple_df, _args())
        assert hasattr(res, "params")
        assert "x" in res.params.index

    def test_ols_slope_approx(self, simple_df):
        res = regress_it(simple_df, _args())
        assert res.params["x"] == pytest.approx(2.0, abs=0.2)

    def test_ols_intercept_approx(self, simple_df):
        res = regress_it(simple_df, _args())
        assert res.params["Intercept"] == pytest.approx(1.0, abs=0.5)

    def test_robust_returns_result(self, simple_df):
        res = regress_it(simple_df, _args(robust=True))
        assert hasattr(res, "params")

    def test_wls_uses_weights(self, weighted_df):
        res = regress_it(weighted_df, _args(formula="y ~ x", weights="w"))
        assert hasattr(res, "params")
        assert "x" in res.params.index

    def test_bad_formula_raises(self, simple_df):
        with pytest.raises(Exception):  # noqa: B017
            regress_it(simple_df, _args(formula="y ~ nonexistent_col"))


# ---------------------------------------------------------------------------
# res2df — output columns
# ---------------------------------------------------------------------------


class TestRes2df:
    def test_ols_columns_present(self, simple_df):
        res = regress_it(simple_df, _args())
        out = res2df(res, _args())
        for col in [
            "tag",
            "variable",
            "marker",
            "effect",
            "stdev",
            "pval",
            "nobs",
            "cilo",
            "cihi",
            "rsquared",
            "condnum",
        ]:
            assert col in out.columns

    def test_robust_no_rsquared(self, simple_df):
        args = _args(robust=True)
        res = regress_it(simple_df, args)
        out = res2df(res, args)
        assert "rsquared" not in out.columns
        assert "condnum" not in out.columns

    def test_one_row_per_predictor(self, simple_df):
        res = regress_it(simple_df, _args())
        out = res2df(res, _args())
        # OLS with intercept: Intercept + x = 2 rows
        assert len(out) == 2

    def test_tag_column_value(self, simple_df):
        args = _args(tag="myrun")
        res = regress_it(simple_df, args)
        out = res2df(res, args)
        assert (out["tag"] == "myrun").all()

    def test_grouped_prepends_group_cols(self, grouped_df):
        args = _args(groupcol=["group"])
        res = regress_it(grouped_df[grouped_df["group"] == "A"], args)
        out = res2df(res, args, groupname="A")
        assert "group" in out.columns
        assert (out["group"] == "A").all()

    def test_marker_contains_x(self, simple_df):
        res = regress_it(simple_df, _args())
        out = res2df(res, _args())
        assert "x" in out["marker"].values


# ---------------------------------------------------------------------------
# FitCommand.execute — grouped
# ---------------------------------------------------------------------------


class TestFitCommandGrouped:
    def test_grouped_output_has_both_groups(self, grouped_df):
        """execute() with groupcol produces rows for each group."""
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".tsv", mode="w", delete=False) as f:
            grouped_df.to_csv(f, sep="\t", index=False)
            fname = f.name
        try:
            cmd = FitCommand()
            args = argparse.Namespace(
                DATAFILE=fname,
                formula="y ~ x",
                weights=None,
                tag="tag",
                robust=False,
                groupcol=["group"],
                summary=False,
                anova=False,
                # io.printdf defaults
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
            )
            buf = _io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = buf
            cmd.execute(args)
            sys.stdout = old_stdout
            output = buf.getvalue()
            assert "A" in output
            assert "B" in output
        finally:
            os.unlink(fname)

    def test_group_regression_slopes(self, grouped_df):
        """Each group's slope should be approximately correct."""
        args = _args(groupcol=["group"])
        parts = []
        for gname, gdf in grouped_df.groupby("group"):
            res = regress_it(gdf, args)
            parts.append(res2df(res, args, groupname=gname))
        out = pd.concat(parts, ignore_index=True)

        x_rows = out[out["marker"] == "x"]
        slope_a = x_rows[x_rows["group"] == "A"]["effect"].iloc[0]
        slope_b = x_rows[x_rows["group"] == "B"]["effect"].iloc[0]
        assert slope_a == pytest.approx(3.0, abs=0.5)
        assert slope_b == pytest.approx(-1.0, abs=0.5)


# ---------------------------------------------------------------------------
# FitCommand.execute — summary / anova modes
# ---------------------------------------------------------------------------


class TestFitCommandSummaryAnova:
    def _run_execute(self, grouped_df_or_simple, args):
        import os
        import tempfile

        df = grouped_df_or_simple
        with tempfile.NamedTemporaryFile(suffix=".tsv", mode="w", delete=False) as f:
            df.to_csv(f, sep="\t", index=False)
            fname = f.name
        args.DATAFILE = fname
        buf = _io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            FitCommand().execute(args)
        finally:
            sys.stdout = old_stdout
            os.unlink(fname)
        return buf.getvalue()

    def _base_args(self, **kwargs):
        base = argparse.Namespace(
            DATAFILE=None,
            formula="y ~ x",
            weights=None,
            tag="tag",
            robust=False,
            groupcol=None,
            summary=False,
            anova=False,
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
        )
        for k, v in kwargs.items():
            setattr(base, k, v)
        return base

    def test_summary_prints_text(self, simple_df):
        args = self._base_args(summary=True)
        output = self._run_execute(simple_df, args)
        assert "OLS" in output or "Dep. Variable" in output

    def test_anova_prints_table(self, simple_df):
        args = self._base_args(anova=True)
        output = self._run_execute(simple_df, args)
        assert "df" in output or "PR(>F)" in output

    def test_default_output_is_tabular(self, simple_df):
        args = self._base_args()
        output = self._run_execute(simple_df, args)
        assert "marker" in output
        assert "effect" in output


# ---------------------------------------------------------------------------
# FitCommand — column validation
# ---------------------------------------------------------------------------


class TestFitCommandValidation:
    def test_missing_groupcol_raises(self, simple_df):
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".tsv", mode="w", delete=False) as f:
            simple_df.to_csv(f, sep="\t", index=False)
            fname = f.name
        try:
            args = argparse.Namespace(
                DATAFILE=fname,
                formula="y ~ x",
                weights=None,
                tag="tag",
                robust=False,
                groupcol=["no_such_col"],
                summary=False,
                anova=False,
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
            )
            with pytest.raises(ValueError, match="no_such_col"):
                FitCommand().execute(args)
        finally:
            os.unlink(fname)

    def test_missing_weights_raises(self, simple_df):
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".tsv", mode="w", delete=False) as f:
            simple_df.to_csv(f, sep="\t", index=False)
            fname = f.name
        try:
            args = argparse.Namespace(
                DATAFILE=fname,
                formula="y ~ x",
                weights="no_such_weight",
                tag="tag",
                robust=False,
                groupcol=None,
                summary=False,
                anova=False,
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
            )
            with pytest.raises(ValueError, match="no_such_weight"):
                FitCommand().execute(args)
        finally:
            os.unlink(fname)
